"""会話の記録を引く道具を、一時フォルダの記録の見本で、公開 CLI から検証する。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[5]
TOOL = REPO / "reports/tools/chat_utterances.py"


def claude_user(timestamp, text, **extra):
    """Claude Code の形の、開発者の発言の行を作る。"""
    row = {"type": "user", "isSidechain": False, "sessionId": "claude-session-1",
           "entrypoint": "claude-desktop", "timestamp": timestamp,
           "message": {"role": "user", "content": text}}
    row.update(extra)
    return row


def claude_tools(timestamp, count):
    """Claude Code の形の、AI が道具を count 回使った応答の行を作る。"""
    uses = [{"type": "tool_use", "id": f"t{i}", "name": "Read", "input": {}} for i in range(count)]
    return {"type": "assistant", "isSidechain": False, "sessionId": "claude-session-1",
            "timestamp": timestamp,
            "message": {"role": "assistant", "content": [{"type": "text", "text": "答え"}, *uses]}}


class ChatUtterancesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "records"
        self.root.mkdir()

    def write(self, name, rows):
        """記録の見本を書く。文字列の行は、壊れた行としてそのまま書く。"""
        dest = self.root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines = [row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
                 for row in rows]
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-B", str(TOOL), *args],
                              capture_output=True, encoding="utf-8")

    def utterances(self, *args):
        """発言の行と、末尾の飛ばした数の行を分けて返す。"""
        result = self.run_tool("--root", str(self.root), *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        return rows[:-1], rows[-1]

    def test_claude_record_returns_only_developer_utterances(self):
        # 道具の結果・環境が足した行・通知を混ぜ、開発者の発言だけが返ることを通す。
        self.write("project/claude.jsonl", [
            {"type": "mode", "mode": "default"},
            claude_user("2026-11-04T03:00:00.000Z", "TempoRL はなぜ効いたのか"),
            claude_tools("2026-11-04T03:00:05.000Z", 1),
            claude_user("2026-11-04T03:00:06.000Z", [
                {"type": "tool_result", "tool_use_id": "t0", "content": "TempoRL の結果"}]),
            claude_tools("2026-11-04T03:00:07.000Z", 1),
            claude_user("2026-11-04T03:00:08.000Z", "TempoRL の手順", isMeta=True),
            claude_user("2026-11-04T03:00:09.000Z", "TempoRL の通知",
                        origin={"kind": "task-notification"}),
            claude_user("2026-11-04T03:00:10.000Z", "TempoRL の依頼", entrypoint="sdk-cli"),
            claude_user("2026-11-04T03:00:11.000Z", "TempoRL の要約", isCompactSummary=True),
            claude_user("2026-11-04T03:00:12.000Z",
                        "<task-notification>TempoRL の完了</task-notification>"),
            claude_user("2026-11-04T03:00:13.000Z", "<system-reminder>札</system-reminder>\n"
                        "<side-sessions-event>TempoRL の会話が終わった</side-sessions-event>"),
            claude_user("2026-11-04T03:01:00.000Z", [{"type": "text", "text":
                        "<system-reminder>TempoRL の札</system-reminder>batch の話"}]),
            claude_tools("2026-11-04T03:01:05.000Z", 1),
            claude_user("2026-11-04T03:02:00.000Z",
                        "<command-message>anet-query-wiki</command-message>\n"
                        "<command-name>/anet-query-wiki</command-name>\n"
                        "<command-args>TempoRL の教訓は何か</command-args>"),
        ])
        rows, tail = self.utterances("--word", "temporl")
        self.assertEqual(rows, [
            {"date": "2026-11-04", "session": "claude-session-1", "source": "claude",
             "tools": 2, "text": "TempoRL はなぜ効いたのか"},
            {"date": "2026-11-04", "session": "claude-session-1", "source": "claude",
             "tools": 0, "text": "/anet-query-wiki TempoRL の教訓は何か"}])
        self.assertEqual(tail, {"skipped_files": 0, "skipped_lines": 0})

    def test_codex_record_returns_only_main_conversation_utterances(self):
        # 人が始めた会話の発言だけが返り、サブエージェントの会話の依頼は返らない。
        def user_message(timestamp, text):
            return {"timestamp": timestamp, "type": "event_msg", "payload": {
                "type": "item_completed",
                "item": {"type": "UserMessage", "id": "u", "content": [
                    {"type": "text", "text": text}, {"type": "local_image", "path": "x"}]}}}

        def response(timestamp, kind, **fields):
            return {"timestamp": timestamp, "type": "response_item",
                    "payload": {"type": kind, **fields}}

        self.write("2026/11/10/rollout-main.jsonl", [
            {"timestamp": "2026-11-10T01:00:00.000Z", "type": "session_meta",
             "payload": {"id": "codex-session-1", "source": "vscode"}},
            response("2026-11-10T01:00:00.500Z", "message", role="user", content=[
                {"type": "input_text", "text": "<environment_context>RND</environment_context>"}]),
            user_message("2026-11-10T01:00:01.000Z", "RND のリセットの事例はあるか"),
            response("2026-11-10T01:00:02.000Z", "custom_tool_call", name="exec", input="ls"),
            response("2026-11-10T01:00:03.000Z", "custom_tool_call_output", output="RND"),
            response("2026-11-10T01:00:04.000Z", "function_call", name="send", arguments="{}"),
            response("2026-11-10T01:00:05.000Z", "function_call_output", output="ok"),
            user_message("2026-11-10T01:02:00.000Z", "<send_user_message_question_reply>"
                         "RND の選択肢 1</send_user_message_question_reply>"),
            response("2026-11-10T01:02:01.000Z", "custom_tool_call", name="exec", input="ls"),
            user_message("2026-11-10T01:05:00.000Z", "次の話"),
            response("2026-11-10T01:05:01.000Z", "custom_tool_call", name="exec", input="ls"),
        ])
        self.write("2026/11/10/rollout-sub.jsonl", [
            {"timestamp": "2026-11-10T02:00:00.000Z", "type": "session_meta",
             "payload": {"id": "codex-sub-1", "source": {"subagent": {"other": "guardian"}}}},
            user_message("2026-11-10T02:00:01.000Z", "RND の変更を見直す"),
        ])
        self.write("2026/11/10/rollout-exec.jsonl", [
            {"timestamp": "2026-11-10T03:00:00.000Z", "type": "session_meta",
             "payload": {"id": "codex-exec-1", "source": "exec"}},
            user_message("2026-11-10T03:00:01.000Z", "RND のレビューの依頼"),
        ])
        rows, tail = self.utterances("--word", "RND")
        # 質問欄への返答は発言に数えず、道具を数える区間も切り替えない。
        self.assertEqual(rows, [{"date": "2026-11-10", "session": "codex-session-1",
                                 "source": "codex", "tools": 3,
                                 "text": "RND のリセットの事例はあるか"}])
        self.assertEqual(tail, {"skipped_files": 0, "skipped_lines": 0})

    def write_boundary_days(self):
        """日本時間の日付の境目の前後に、発言を 1 つずつ置く。"""
        self.write("project/days.jsonl", [
            claude_user("2026-11-03T14:59:59.000Z", "前の日の lane 窓"),
            claude_user("2026-11-03T15:00:00.000Z", "始まりの日の batch"),
            claude_tools("2026-11-03T15:00:01.000Z", 3),
            claude_user("2026-11-05T14:59:59.000Z", "終わりの日の lane 窓"),
            claude_user("2026-11-05T15:00:00.000Z", "次の日の lane 窓"),
        ])

    def test_period_includes_both_end_days_in_japan_time(self):
        self.write_boundary_days()
        rows, _ = self.utterances("--since", "2026-11-04", "--until", "2026-11-05")
        self.assertEqual([(row["date"], row["tools"], row["text"]) for row in rows],
                         [("2026-11-04", 3, "始まりの日の batch"),
                          ("2026-11-05", 0, "終わりの日の lane 窓")])

    def test_word_and_period_narrow_each_other(self):
        self.write_boundary_days()
        rows, _ = self.utterances("--word", "lane 窓", "--since", "2026-11-04",
                                  "--until", "2026-11-05")
        self.assertEqual([row["text"] for row in rows], ["終わりの日の lane 窓"])

    def test_unknown_files_and_broken_lines_are_counted_as_skipped(self):
        # 壊れた行と、どちらの形でもないファイルを混ぜても、ほかの行は返る。
        self.write("project/claude.jsonl", [
            claude_user("2026-11-04T03:00:00.000Z", "NoisyNet の前の話"),
            "{壊れた行 NoisyNet",
            "[1, 2]",
            claude_user("2026-11-04T04:00:00.000Z", "NoisyNet のあとの話"),
        ])
        self.write("project/other.jsonl", [{"note": "NoisyNet の記録ではない"}])
        self.write("project/empty.jsonl", [])
        rows, tail = self.utterances("--word", "NoisyNet")
        self.assertEqual([row["text"] for row in rows],
                         ["NoisyNet の前の話", "NoisyNet のあとの話"])
        self.assertEqual(tail, {"skipped_files": 2, "skipped_lines": 2})

    def test_invalid_arguments_fail_without_output(self):
        # 語も期間も無い、期間が読めない、期間が逆、置き場が無い、のどれも行を出さない。
        self.write_boundary_days()
        root = str(self.root)
        cases = {
            "語も期間も無い": ["--root", root],
            "期間の片方だけ": ["--root", root, "--since", "2026-11-04"],
            "日付として読めない": ["--root", root, "--since", "2026-11-04", "--until", "11/05"],
            "存在しない日付": ["--root", root, "--since", "2026-11-04", "--until", "2026-11-31"],
            "始まりが終わりより後": ["--root", root, "--since", "2026-11-05",
                                     "--until", "2026-11-04"],
            "置き場が無い": ["--root", str(self.root / "missing"), "--word", "lane"],
        }
        for name, args in cases.items():
            with self.subTest(name):
                result = self.run_tool(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
