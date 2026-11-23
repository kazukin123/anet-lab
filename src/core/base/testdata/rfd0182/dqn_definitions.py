import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

# コメントと文字列を1トークンとして読んで、中にある括弧を構文から除く。
TOKEN = re.compile(
    r"(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|"
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)")|'
    r'(?P<string>"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\')|'
    r"(?P<directive>^[ \t]*#(?:[^\n\\]|\\[^\n]|\\\n)*)|"
    r"(?P<word>[A-Za-z_]\w*|\d[\w'.]*)|(?P<punctuation>::|->|&&|\|\||[^\s])",
    re.MULTILINE,
)

@dataclass
class Unit:
    start: int
    end: int
    tokens: list

    @property
    def title(self):
        result = []
        for token in self.tokens[:50]:
            if token[0] in ("{", ";") and result:
                break
            result.append(token[0])
        return " ".join(result)

def units(text):
    tokens = [(match[0], match.start(), match.end(), match.lastgroup)
              for match in TOKEN.finditer(text) if match.lastgroup != "comment"]
    return list(parse_units(tokens, 0, len(tokens)))

def close_block(tokens, start, end):
    depth = 1
    for index in range(start + 1, end):
        if tokens[index][0] == "{":
            depth += 1
        elif tokens[index][0] == "}":
            depth -= 1
            if not depth:
                return index
    raise ValueError(f"Unclosed block at offset {tokens[start][1]}")

def parse_units(tokens, start, end):
    begin = start
    index = start
    parens = 0
    brackets = 0
    while index < end:
        token, _, _, kind = tokens[index]
        if index == begin and kind == "directive":
            yield Unit(tokens[index][1], tokens[index][2], [tokens[index]])
            index += 1
            begin = index
            continue
        if token == "(":
            parens += 1
        elif token == ")":
            parens -= 1
        elif token == "[":
            brackets += 1
        elif token == "]":
            brackets -= 1
        if token == "{" and parens == brackets == 0:
            close = close_block(tokens, index, end)
            if tokens[begin][0] == "namespace":
                yield from parse_units(tokens, index + 1, close)
                index = close + 1
                begin = index
                continue
            index = close
            # class、初期化子、try/catch には、閉じ括弧の後にも構成要素が続く。
            if index + 1 < end and tokens[index + 1][0] in (";", ",", "catch", "{"):
                index += 1
                continue
            yield Unit(tokens[begin][1], tokens[index][2], tokens[begin:index + 1])
            begin = index + 1
        elif token == ";" and parens == brackets == 0:
            yield Unit(tokens[begin][1], tokens[index][2], tokens[begin:index + 1])
            begin = index + 1
        index += 1
    if begin != end:
        raise ValueError(f"Unclosed declaration at offset {tokens[begin][1]}")

def definitions(path, test_cases=False):
    text = path.read_text(encoding="utf-8-sig")
    result = Counter()
    for unit in units(text):
        words = [token[0] for token in unit.tokens]
        if not words or unit.tokens[0][3] == "directive":
            continue
        if test_cases and not words[0].startswith("TEST_CASE"):
            continue
        if "{" not in words and ("=" not in words or words[0] in {"namespace", "using"}):
            continue
        while words[0] in {"static", "extern", "inline"}:
            words.pop(0)
        # 定義の本体を変えず、外から呼べる名前空間へ移した宣言の修飾を除く。
        split = words.index("{") if "{" in words else len(words)
        signature = " ".join(words[:split]).replace("anet :: rl :: dqn :: ", "").split()
        result[tuple(signature + words[split:])] += 1
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--after", nargs="+", type=Path)
    parser.add_argument("--test-cases", action="store_true")
    args = parser.parse_args()
    if args.after:
        old = sum((definitions(path, args.test_cases) for path in args.paths), Counter())
        new = sum((definitions(path, args.test_cases) for path in args.after), Counter())
        removed = old - new
        added = new - old
        print(f"Compared {old.total()} -> {new.total()} definitions. Removed: {removed.total()}. Added: {added.total()}.")
        for label, changes in (("Removed", removed), ("Added", added)):
            for key, count in changes.items():
                print(label, count, " ".join(key[:20]))
        raise SystemExit(bool(removed or added))
    for path in args.paths:
        text = path.read_text(encoding="utf-8-sig")
        for unit in units(text):
            if unit.tokens[0][3] != "directive":
                line = text.count("\n", 0, unit.start) + 1
                print(f"{line:5} {unit.title}")

if __name__ == "__main__":
    main()
