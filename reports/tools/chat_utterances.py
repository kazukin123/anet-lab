"""会話の記録から、開発者の発言の行だけを引く（読み取りだけ）。

使い方:
    chat_utterances.py --root <記録の置き場> [--root ...] [--word <語> ...]
                       [--since YYYY-MM-DD --until YYYY-MM-DD]

- 置き場の下の *.jsonl を再帰で読む。Claude Code の記録と Codex の記録を、中身で見分ける。
- 語と期間は、どちらかが要る。両方を渡すと、期間の中で語に当たる発言を返す。
  期間は始まりの日と終わりの日を含み、日付は記録の時刻を日本時間に直して数える。
- 出力は 1 行 1 発言の JSON（date・session・source・tools・text）で、tools は、
  その発言のあと次の開発者の発言までに AI が道具を使った回数である。
  最後の 1 行は、飛ばしたファイルと行の数である。
- 残さないと断りのある会話を外す判定と、Plan Mode の会話かどうかの判定は、読む側が行う。
"""
import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys


JST = timezone(timedelta(hours=9))
# 環境が発言に差し込む札と、貼り付けの中身。開発者の言葉ではないので除く。
INJECTED = re.compile(
    r"(?s)<(system-reminder|pasted_content|local-command-stdout|local-command-caveat"
    r"|environment_context|ide_selection|ide_opened_file|command-message)\b[^>]*>"
    r".*?</\1[^>]*>")
COMMAND_NAME = re.compile(r"(?s)<command-name>(.*?)</command-name>")
COMMAND_ARGS = re.compile(r"(?s)<command-args>(.*?)</command-args>")
# 人が文として打った発言でない行の書き出し（通知、ほかの会話の出来事、質問欄への返答）。
NOT_UTTERANCE = ("<task-notification", "<side-sessions-event", "<ci-monitor-event",
                 "<send_user_message_question_reply")


def jst_date(timestamp):
    """記録の時刻を日本時間の日付に直す。読めなければ None を返す。"""
    try:
        moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(JST).date().isoformat()


def clean(text):
    """発言の文から、環境が差し込んだ部分を除く。コマンドは「/名前 引数」に直す。

    人が文として打った発言でなければ、None を返す。
    """
    text = INJECTED.sub("", text).strip()
    if text.startswith(NOT_UTTERANCE):
        return None
    name = COMMAND_NAME.search(text)
    if name:
        args = COMMAND_ARGS.search(text)
        command = " ".join(part.strip() for part in (name.group(1), args.group(1) if args else "")
                           if part.strip())
        text = command + "\n" + COMMAND_ARGS.sub("", COMMAND_NAME.sub("", text))
    return text.strip()


def claude_utterance(entry):
    """Claude Code の user の行が開発者の発言なら、その文を返す。違えば None を返す。"""
    # 環境が足した行、要約、人の操作でない起動は、開発者の発言ではない。
    if entry.get("isMeta") or entry.get("isCompactSummary") \
            or str(entry.get("entrypoint", "")).startswith("sdk-"):
        return None
    # 通知と、ほかの会話からの発言は、出どころの欄で分かる。
    origin = entry.get("origin")
    if isinstance(origin, dict) and origin.get("kind") != "human":
        return None
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, list):
        # 道具の結果は user の行として記録される。
        parts = [part for part in content if isinstance(part, dict)]
        if any(part.get("type") == "tool_result" for part in parts):
            return None
        content = "\n".join(part.get("text", "") for part in parts if part.get("type") == "text")
    if not isinstance(content, str):
        return None
    return clean(content)


def read_claude(entries, fallback_session):
    """Claude Code の形の行から、発言と、そのあとの道具の回数を組にして返す。"""
    found = []
    current = None
    for entry in entries:
        if entry.get("isSidechain"):
            continue
        if entry.get("type") == "user":
            text = claude_utterance(entry)
            if text is None:
                continue
            # 次の開発者の発言で、道具を数える区間を切り替える。
            current = {"date": jst_date(entry.get("timestamp")),
                       "session": entry.get("sessionId") or fallback_session,
                       "source": "claude", "tools": 0, "text": text}
            found.append(current)
        elif entry.get("type") == "assistant" and current is not None:
            content = (entry.get("message") or {}).get("content")
            if isinstance(content, list):
                current["tools"] += sum(1 for part in content if isinstance(part, dict)
                                        and part.get("type") == "tool_use")
    return found


def read_codex(entries, fallback_session):
    """Codex の形の行から、発言と、そのあとの道具の回数を組にして返す。"""
    meta = entries[0].get("payload") if isinstance(entries[0].get("payload"), dict) else {}
    # サブエージェントの会話と、端末から依頼を流し込んだ会話は、開発者の発言を持たない。
    if not isinstance(meta.get("source"), str) or meta["source"] == "exec":
        return []
    found = []
    current = None
    for entry in entries[1:]:
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        kind = str(payload.get("type", ""))
        item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
        if entry.get("type") == "event_msg" and kind == "item_completed" \
                and item.get("type") == "UserMessage":
            parts = item.get("content") if isinstance(item.get("content"), list) else []
            text = clean("\n".join(part.get("text", "") for part in parts
                                   if isinstance(part, dict) and part.get("type") == "text"))
            if text is None:
                continue
            # 次の開発者の発言で、道具を数える区間を切り替える。
            current = {"date": jst_date(entry.get("timestamp")),
                       "session": meta.get("id") or fallback_session,
                       "source": "codex", "tools": 0, "text": text}
            found.append(current)
        elif entry.get("type") == "response_item" and kind.endswith("_call") \
                and current is not None:
            current["tools"] += 1
    return found


def read_file(path):
    """記録 1 ファイルを読み、発言の一覧と、壊れた行の数を返す。

    どちらの環境の形にも当たらないファイルでは、None を返す。
    """
    entries = []
    broken = 0
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                entry = None
            # JSON として読めない行と、オブジェクトでない行は、壊れた行として数える。
            if isinstance(entry, dict):
                entries.append(entry)
            else:
                broken += 1
    # 先頭の行で Codex の記録を、発言か応答の行の有無で Claude Code の記録を見分ける。
    if entries and entries[0].get("type") == "session_meta":
        return read_codex(entries, path.stem), broken
    if any(entry.get("type") in ("user", "assistant") and isinstance(entry.get("message"), dict)
           for entry in entries):
        return read_claude(entries, path.stem), broken
    return None


def parse_args():
    """引数を読み、出力を始める前に全部を検査する。違反は終了コード 2 で止める。"""
    parser = argparse.ArgumentParser(description="List developer utterances in chat records.")
    parser.add_argument("--root", action="append", required=True,
                        help="directory that holds chat records (*.jsonl, searched recursively)")
    parser.add_argument("--word", action="append", default=[],
                        help="keep utterances that contain this word (repeatable, any match)")
    parser.add_argument("--since", help="first day of the period, YYYY-MM-DD (Japan time)")
    parser.add_argument("--until", help="last day of the period, YYYY-MM-DD (Japan time)")
    args = parser.parse_args()

    if not args.word and args.since is None and args.until is None:
        parser.error("give --word, or --since and --until")
    if (args.since is None) != (args.until is None):
        parser.error("--since and --until must be given together")
    for name in ("since", "until"):
        value = getattr(args, name)
        try:
            # 0 埋めの YYYY-MM-DD だけを受ける。あとで文字列のまま比べるためである。
            if value is not None and date.fromisoformat(value).isoformat() != value:
                raise ValueError(value)
        except ValueError:
            parser.error(f"--{name} is not a date in YYYY-MM-DD: {value}")
    if args.since is not None and args.since > args.until:
        parser.error(f"--since {args.since} is after --until {args.until}")
    for root in args.root:
        if not Path(root).is_dir():
            parser.error(f"--root is not a directory: {root}")
    return args


def main():
    args = parse_args()

    # 記録を読み、期間と語に当たる発言だけを残す。
    words = [word.casefold() for word in args.word]
    rows = []
    skipped = {"skipped_files": 0, "skipped_lines": 0}
    for root in args.root:
        for path in sorted(Path(root).rglob("*.jsonl")):
            result = read_file(path)
            if result is None:
                skipped["skipped_files"] += 1
                continue
            skipped["skipped_lines"] += result[1]
            for row in result[0]:
                # 差し込みを除いて空になった発言と、日付を読めない発言は返さない。
                if not row["text"] or row["date"] is None:
                    continue
                if args.since is not None and not args.since <= row["date"] <= args.until:
                    continue
                if words and not any(word in row["text"].casefold() for word in words):
                    continue
                rows.append(row)

    # 発言の行のあとに、飛ばした数を 1 行で出す。
    sys.stdout.reconfigure(encoding="utf-8")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps(skipped))


if __name__ == "__main__":
    main()
