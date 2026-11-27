"""RFD0202 の wiki 点検を、隔離した作業ツリーで公開 CLI から検証する。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


REPO = Path(__file__).resolve().parents[5]
HELPER = REPO / "reports/tools/wiki_support.py"


class WikiScanTest(unittest.TestCase):
    def setUp(self):
        # 元のリポジトリの index と履歴を使わず、起点を空の tree にする。
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.git("init", "-q")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "fixture")
        self.start = self.git("rev-parse", "HEAD")
        self.write("docs/wiki/log.md", "2026-11-26 start rev:" + self.start + "\n")
        self.write("docs/wiki/index.md", "# index\n")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, env=self.env,
                                       encoding="utf-8").strip()

    def write(self, path, contents):
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(contents, encoding="utf-8", newline="\n")

    def scan(self):
        result = subprocess.check_output([sys.executable, "-B", str(HELPER), "scan"],
                                         cwd=self.root, env=self.env, encoding="utf-8")
        return json.loads(result)

    def test_headers_writes_summary_and_concept_properties(self):
        # 材料・本文・目録・作業の記録から、二種類のページのヘッダを作る。
        summary = "docs/wiki/summary/reports/probe.md"
        concept = "docs/wiki/concepts/methods/probe.md"
        summary_body = "# サマリの題名\n\n## 主要な概念\n\n- [本文での呼び名](../../concepts/methods/probe.md)\n"
        concept_body = "# ページの題名\n"
        self.write("reports/probe.md", "# 原本\n")
        self.write(summary, summary_body)
        self.write(concept, concept_body)
        self.write("docs/wiki/index.md", "- [目録の題名 — 計測](concepts/methods/probe.md) — 説明 （別名: 別名 A, 別名 B）。\n")
        self.write("docs/wiki/log.md", f"2026-11-26 start rev:{self.start}\n"
                   f"2026-11-27 fix 試用 → {summary}, {concept}\n")

        self.headers()

        self.assertEqual((self.root / summary).read_text(encoding="utf-8"),
                         "---\ntitle: サマリの題名\npage_kind: サマリ\n"
                         'material: "[[reports/probe.md|probe]]"\nmaterial_kind: 報告\n'
                         'concepts:\n  - "[[docs/wiki/concepts/methods/probe.md|ページの題名]]"\n'
                         "updated: 2026-11-27\n---\n" + summary_body)
        self.assertEqual((self.root / concept).read_text(encoding="utf-8"),
                         "---\ntitle: ページの題名\npage_kind: 手法\n"
                         "aliases:\n  - 別名 A\n  - 別名 B\ndescription: 説明\n"
                         "updated: 2026-11-27\n---\n" + concept_body)

    def headers(self, check=True):
        return subprocess.run([sys.executable, "-B", str(HELPER), "headers"],
                              cwd=self.root, env=self.env, encoding="utf-8",
                              capture_output=True, check=check)

    def test_headers_replaces_managed_values_and_preserves_custom_yaml_and_body(self):
        page = "docs/wiki/concepts/methods/probe.md"
        dest = self.root / page
        body = b"# Actual title\r\n\r\nbody\r\n"
        custom = "tags: [keep, 'with spaces']\ncustom: |-\n  first line\n\n  second line\n"
        self.write(page, "---\ntitle: Wrong\n" + custom
                   + "\n# Unattached comment\nupdated: 2000-01-01\n---\n")
        dest.write_bytes(dest.read_bytes() + body)
        expected = ("---\ntitle: Actual title\npage_kind: 手法\naliases: []\ndescription:\nupdated:\n"
                    + custom + "---\n").encode("utf-8") + body

        self.headers()
        self.assertEqual(dest.read_bytes(), expected)
        self.assertEqual(json.loads(self.headers().stdout)["updated"], [])
        self.assertEqual(dest.read_bytes(), expected)

    def test_headers_validates_every_page_before_writing(self):
        good = "docs/wiki/concepts/methods/first.md"
        bad = "docs/wiki/concepts/methods/probe.md"
        misplaced = "docs/wiki/summary/unknown/probe.md"
        summary = "docs/wiki/summary/rfd/0200_probe_10rfd.md"
        index = "docs/wiki/index.md"
        log = "docs/wiki/log.md"
        duplicate_entry = "- [p — q](concepts/methods/probe.md) — description\n"
        cases = (
            ({bad: "---\ntitle: broken\n# probe\n"}, bad, "unclosed YAML header"),
            ({bad: "---\ntitle: [broken\n---\n# probe\n"}, bad, "invalid YAML header"),
            ({bad: "---\n- value\n---\n# probe\n"}, bad, "YAML header must be a mapping"),
            ({bad: "## Body\n"}, bad, "missing title"),
            ({misplaced: "# probe\n"}, misplaced, "unrecognized page folder"),
            ({summary: "# probe\n", "docs/memo/0200_probe_10rfd.md": "first",
              "docs/memo/done/0200_probe_10rfd.md": "second"}, summary, "multiple materials"),
            ({bad: "# probe\n", index: duplicate_entry * 2}, bad, "multiple index entries"),
            ({index: None}, index, "index.md"),
            ({log: None}, log, "log.md"),
        )
        for inputs, page, reason in cases:
            with self.subTest(reason=reason):
                self.write(good, "# First\n")
                self.write(index, "# Index\n")
                self.write(log, f"2026-11-26 start rev:{self.start}\n")
                for path, contents in inputs.items():
                    if contents is None:
                        (self.root / path).unlink()
                    else:
                        self.write(path, contents)
                before = {p: p.read_bytes() for p in self.root.joinpath("docs").rglob("*.md")}
                result = self.headers(check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(page, result.stderr.replace(chr(92), "/"))
                self.assertIn(reason, result.stderr)
                self.assertEqual({p: p.read_bytes() for p in self.root.joinpath("docs").rglob("*.md")}, before)
                for path in inputs:
                    (self.root / path).unlink(missing_ok=True)

    def test_headers_retains_quoted_flow_keys_and_multiline_values(self):
        page = "docs/wiki/concepts/methods/probe.md"
        body = "# Probe\n"
        headers = (
            ('{"title":"old","custom":["one", "two"]}\n', '"custom": ["one", "two"]\n'),
            ("title: old\ncustom: |+\n  line\n\n", "custom: |+\n  line\n\n"),
        )
        for original, custom in headers:
            with self.subTest(original=original):
                self.write(page, "---\n" + original + "---\n" + body)
                self.headers()
                text = (self.root / page).read_text(encoding="utf-8")
                self.assertTrue(text.endswith("\n" + custom + "---\n" + body), text)
                self.assertEqual(json.loads(self.headers().stdout)["updated"], [])

    def test_headers_preserves_custom_explicit_keys_and_anchor_references(self):
        page = "docs/wiki/concepts/methods/probe.md"
        body = "# Current title\n"
        custom = "memo: *label\n?\n  custom\n: value\n"
        self.write(page, "---\ntitle: &label Old title\n" + custom + "---\n" + body)
        self.headers()
        text = (self.root / page).read_text(encoding="utf-8")
        header, rest = text.split("\n---\n", 1)
        self.assertIn("\n" + custom, text)
        self.assertEqual(rest, body)
        decoded = yaml.safe_load(header)
        self.assertEqual(decoded["title"], "Current title")
        self.assertEqual(decoded["memo"], "Old title")
        self.assertEqual(decoded["custom"], "value")
        self.assertEqual(json.loads(self.headers().stdout)["updated"], [])

    def test_headers_preserves_anchor_dependencies_and_merges_without_duplicate_keys(self):
        page = "docs/wiki/concepts/methods/probe.md"
        body = "# Current title\n"
        custom = "memo: &base Old title\n"
        references = "extra: *label\n"
        for merge in ("", "<<: {from_merge: 9}\n"):
            with self.subTest(merge=merge):
                self.write(page, "---\n" + custom + "title: &label [*base]\n" + merge
                           + references + "---\n" + body)
                self.headers()
                text = (self.root / page).read_text(encoding="utf-8")
                header, rest = text.split("\n---\n", 1)
                self.assertIn("\n" + custom, text)
                self.assertIn("\n" + references, text)
                self.assertEqual(rest, body)
                self.assertEqual(header.count("\n<<:"), 1)
                decoded = yaml.safe_load(header)
                self.assertEqual(decoded["title"], "Current title")
                self.assertEqual(decoded["memo"], "Old title")
                self.assertEqual(decoded["extra"], ["Old title"])
                if merge:
                    self.assertEqual(decoded["from_merge"], 9)
                self.assertEqual(json.loads(self.headers().stdout)["updated"], [])

    def test_scan_reports_header_mismatches_and_page_errors_without_enabling_lint(self):
        # 直せるヘッダと、入力を直す必要があるページを同時に報告する。
        folder = "docs/wiki/concepts/methods/"
        good = folder + "good.md"
        stale = folder + "stale.md"
        self.write(good, "# Good\n")
        self.write(stale, "# Stale\n")
        self.headers()
        path = self.root / stale
        self.write(stale, path.read_text(encoding="utf-8").replace("title: Stale", "title: Wrong"))
        missing = folder + "missing.md"
        self.write(missing, "# Missing\n")
        errors = {
            folder + "unclosed.md": ("---\n# Page\n", "unclosed YAML header"),
            folder + "invalid.md": ("---\ntitle: [broken\n---\n# Page\n", "invalid YAML header"),
            folder + "not_a_mapping.md": ("---\n- item\n---\n# Page\n", "YAML header must be a mapping"),
            folder + "no_title.md": ("## Only a section\n", "missing title"),
            "docs/wiki/concepts/unknown/probe.md": ("# Page\n", "unrecognized page folder"),
            folder + "duplicate_index.md": ("# Page\n", "multiple index entries"),
        }
        for filename, (body, _) in errors.items():
            self.write(filename, body)
        entry = "- [page](concepts/methods/duplicate_index.md) — description\n"
        self.write("docs/wiki/index.md", entry * 2)
        result = self.scan()
        warnings = {w["page"]: w["reason"] for w in result["warnings"] if w["rule"] == "YAML ヘッダ"}
        self.assertFalse(result["eligible"])
        self.assertEqual(set(warnings), {stale, missing, *errors})
        for filename, (_, reason) in errors.items():
            self.assertIn(reason, warnings[filename])

        summary = "docs/wiki/summary/rfd/0200_probe_10rfd.md"
        self.write(summary, "# Page\n")
        self.write("docs/memo/0200_probe_10rfd.md", "# First material\n")
        self.write("docs/memo/done/0200_probe_10rfd.md", "# Second material\n")
        result = self.scan()
        warnings = {w["page"]: w["reason"] for w in result["warnings"] if w["rule"] == "YAML ヘッダ"}
        self.assertIn("multiple materials", warnings[summary])
        self.assertTrue(result["eligible"])

    def test_headers_copies_latest_log_entry_and_allows_empty_yaml(self):
        page = "docs/wiki/concepts/methods/probe.md"
        body = '# Topic: "a" and \\b\n'
        for header in ("", "# No properties\n\n", "~\n", "null\n"):
            with self.subTest(header=header):
                self.write(page, "---\n" + header + "---\n" + body)
                self.write("docs/wiki/log.md", f"2026-11-26 start rev:{self.start}\n"
                           f"2026-11-26 ingest reports/a.md → {page}（最初の記録）\n"
                           f"2026-11-27 query rfd → docs/wiki/index.md、{page}（追記）\n"
                           f"2026-11-28 fix {page} → なし\n")
                warnings = [w for w in self.scan()["warnings"]
                            if w["page"] == page and w["rule"] == "YAML ヘッダ"]
                self.assertEqual([w["reason"] for w in warnings], ["YAML header differs from generated content"])
                self.headers()
                expected = ('---\ntitle: "Topic: \\"a\\" and \\\\b"\npage_kind: 手法\naliases: []\n'
                            "description:\nupdated: 2026-11-27\n---\n" + body)
                self.assertEqual((self.root / page).read_text(encoding="utf-8"), expected)

        with (self.root / "docs/wiki/log.md").open("a", encoding="utf-8") as out:
            out.write(f"2026-11-29 fix 説明 → {page}\n")
        self.headers()
        self.assertEqual((self.root / page).read_text(encoding="utf-8"), expected.replace("2026-11-27", "2026-11-29"))

    def test_scan_ignores_yaml_when_checking_body_length_format_and_sources(self):
        # 本文の警告や blob はヘッダを入れる前と同じ結果を返す。
        material = "reports/probe.md"
        summary = "docs/wiki/summary/reports/probe.md"
        concept = "docs/wiki/concepts/methods/probe.md"
        self.write(material, "# Original\n")
        blob = self.git("hash-object", "--", material)
        self.write("docs/wiki/log.md", f"2026-11-26 start rev:{self.start}\n"
                   f"2026-11-26 ingest {material} blob:{blob} → {summary}\n")
        body = f"# Summary\n\n## 元の材料\n\nblob:{blob}\n\n## 内容\n\n"
        self.write(summary, body + "字" * (8000 - len(body)))
        self.write(concept, "# Concept\n\n## 今の結論\n\n" + "字" * 201 + "\n"
                   "\n## 関連\n\n- [リンク切れ](missing.md)\n")
        before = self.scan()

        self.headers()
        page = self.root / concept
        prefix, body = page.read_text(encoding="utf-8").split("\n---\n", 1)
        self.write(concept, prefix + '\ncustom: "- [例](missing-in-yaml.md)"\n---\n' + body)
        after = self.scan()
        self.assertEqual(after["pending"], before["pending"])
        self.assertEqual(after["pending"], [])
        self.assertEqual(after["broken_links"], before["broken_links"])
        self.assertEqual([w for w in after["warnings"] if w["rule"] != "YAML ヘッダ"],
                         [w for w in before["warnings"] if w["rule"] != "YAML ヘッダ"])

    def test_scan_reports_links_from_summary_that_do_not_point_to_concept_pages(self):
        summary = "docs/wiki/summary/reports/probe.md"
        concept = "docs/wiki/concepts/methods/concept.md"
        missing = "../../concepts/methods/missing.md"
        not_concept = "../../../wiki/index.md"
        external = "https://example.org/not-a-concept"
        self.write(concept, "# Concept\n")
        self.write(summary, "# Summary\n\n## 主要な概念\n\n"
                   "- [ok](../../concepts/methods/concept.md#section)\n"
                   f"- [missing]({missing})\n- [index]({not_concept})\n"
                   f"- [external]({external})\n"
                   "\n## Other\n\n- [outside section](../../concepts/missing.md)\n")

        self.headers()
        header = (self.root / summary).read_text(encoding="utf-8").split("\n---\n", 1)[0]
        self.assertIn('concepts:\n  - "[[docs/wiki/concepts/methods/concept.md|Concept]]"\nupdated:', header)
        result = self.scan()
        self.assertFalse(result["eligible"])
        self.assertEqual([w["target"] for w in result["warnings"]
                          if w["page"] == summary and w["rule"] == "主要な概念のリンク"],
                         [missing, not_concept, external])

    def test_recorded_material_still_needs_its_summary(self):
        # 作業の記録だけそろった状態から、サマリができるまでを通す。
        material = "reports/probe.md"
        self.write(material, "# probe\n")
        blob = self.git("hash-object", "--", material)
        self.write("docs/wiki/log.md", f"2026-11-26 start rev:{self.start}\n"
                   f"2026-11-26 writeback {material} blob:{blob} → なし\n")
        self.assertEqual([row["path"] for row in self.scan()["pending"]], [material])
        self.write("docs/wiki/summary/reports/probe.md",
                   f"## 元の材料\n\n[probe](../../../../reports/probe.md) blob:{blob}\n")
        self.assertEqual(self.scan()["pending"], [])


    def test_summary_blob_must_match_the_source_section(self):
        # 同じ blob の書き戻しがあっても、元の材料の節が古ければ拾う。
        material = "docs/memo/done/0200_probe_10rfd.md"
        self.write(material, "# old\n")
        old = self.git("hash-object", "--", material)
        self.write(material, "# new\n")
        new = self.git("hash-object", "--", material)
        self.write("docs/wiki/log.md", f"2026-11-26 start rev:{self.start}\n"
                   f"2026-11-26 writeback {material} blob:{new} → なし\n")
        summary = "docs/wiki/summary/rfd/0200_probe_10rfd.md"
        for stored in (old, "", "bogus", new + "0"):
            with self.subTest(stored=stored):
                self.write(summary, f"## 元の材料\n\nblob:{stored}\n\n## 内容\n\nblob:{new}\n")
                self.assertEqual([r["path"] for r in self.scan()["pending"]], [material])
        self.write(summary, f"## 元の材料\n\nblob:{new}\n")
        self.assertEqual(self.scan()["pending"], [])


    def test_orphan_summary_warns_without_enabling_lint(self):
        summary = "docs/wiki/summary/experiments/a/probe.md"
        self.write(summary, "# probe\n")
        result = self.scan()
        self.assertFalse(result["eligible"])
        self.assertIn(summary, [row["page"] for row in result.get("warnings", [])])


    def test_warns_only_for_broken_incoming_wiki_links(self):
        # 引用例を含む未追跡の文書から、移動の前後で外から入るリンクを調べる。
        page = "docs/wiki/concepts/methods/probe.md"
        self.write(page, "# probe\n")
        ref = "docs/notes/entry.md"
        link = "[probe](../wiki/concepts/methods/probe.md#anchor)"
        self.write(ref, link + "\n" + chr(96) + "[example](../wiki/missing.md)" + chr(96))
        self.assertEqual([w for w in self.scan()["warnings"] if w["page"] == ref], [])
        moved = "docs/wiki/concepts/phenomena/probe.md"
        (self.root / moved).parent.mkdir(parents=True)
        (self.root / page).rename(self.root / moved)
        result = self.scan()
        rows = [w for w in result["warnings"] if w["page"] == ref]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidates"], [moved])
        self.assertFalse(result["eligible"])
        self.write(ref, link.replace("methods", "phenomena"))
        self.assertEqual([w for w in self.scan()["warnings"] if w["page"] == ref], [])


    def test_scan_lists_concepts_and_reports_format_warnings(self):
        # 見出し・出典・状態・長さが異なる不備として返ることを確かめる。
        page = "docs/wiki/concepts/subjects/probe.md"
        self.write(page, "## 今の結論\n\n" + "字" * 201 + "\n\n## 分かっていること\n"
                   "\n### 外の事例\n\n## 横断的知見\n\n- 共通点: test（出典: [a](#a)）\n"
                   "\n## 見立て\n\n- 仮説。状態: 保留\n")
        self.write("docs/wiki/index.md", "- [probe](concepts/subjects/probe.md) — "
                   + "字" * 201 + " （別名: test）\n")
        self.write("docs/wiki/summary/adr/large.md", "字" * 8001)
        result = self.scan()
        self.assertIn(page, result["changed_pages"])
        rules = {w["rule"] for w in result["warnings"]}
        for rule in ("知見の節", "知見の小見出し", "今の結論の長さ", "横断的知見の出典",
                     "見立ての状態", "目録の説明の長さ", "サマリの長さ"):
            self.assertIn(rule, rules)
        self.assertFalse(result["eligible"])


    def test_maps_regenerates_framework_and_preserves_external_examples(self):
        # maps が新しい置き場の生成区画だけを書き換えることを確かめる。
        self.write("reports/tools/categories.json", '{"categories": []}')
        self.write("docs/management/issues/INDEX.md", "# index\n")
        self.write("docs/memo/README.md", "# memo\n")
        page = "docs/wiki/concepts/framework/311_dev_tools.md"
        self.write(page, "# map\n\n## 変えるときの条件\n\n## 外の事例\n\nkeep\n\n"
                   "## ファイル\nold\n\n## RFD の一覧\nold\n\n"
                   "## 課題\n<!-- issues:begin -->\nold\n<!-- issues:end -->\n")
        subprocess.run([sys.executable, "-B", str(HELPER), "maps"], cwd=self.root,
                       env=self.env, capture_output=True, check=True)
        body = (self.root / page).read_text(encoding="utf-8")
        self.assertIn("## 外の事例\n\nkeep\n", body)
        self.assertNotIn("\nold\n", body)


    def test_local_source_link_with_spaces_is_resolved(self):
        self.write("docs/resources/example file.md", "# example\n")
        self.write("docs/wiki/concepts/methods/probe.md", "# probe\n\n## 分かっていること\n"
                   "- 出典 [example](<../../../resources/example file.md#anchor>)\n")
        self.assertEqual(self.scan()["broken_links"], [])


    def test_query_leaves_checkpoint_and_pending_unchanged(self):
        self.write("reports/probe.md", "# probe\n")
        before = self.scan()
        log = self.root / "docs/wiki/log.md"
        with log.open("a", encoding="utf-8") as out:
            out.write("2026-11-26 query 質問 → docs/wiki/index.md（判定: 答えの種がある。根拠: なし）\n"
                      "2026-11-26 query （要旨なし） → なし（判定: 無い。根拠: なし）\n")
        after = self.scan()
        for key in ("start", "checkpoint", "pending", "eligible"):
            self.assertEqual(before[key], after[key])

    def test_moved_material_requires_a_summary_at_the_new_path(self):
        material = "docs/experiments/new/name.md"
        self.write(material, "# material\n")
        blob = self.git("hash-object", "--", material)
        self.write("docs/wiki/log.md", f"2026-11-26 start rev:{self.start}\n"
                   f"2026-11-26 ingest docs/experiments/old/name.md blob:{blob} → なし\n")
        self.assertEqual([r["path"] for r in self.scan()["pending"]], [material])
        self.write("docs/wiki/summary/experiments/new/name.md", f"## 元の材料\n\nblob:{blob}\n")
        result = self.scan()
        self.assertEqual(result["pending"], [])
        self.assertEqual([r["path"] for r in result["moved"]], [material])


    def test_modified_original_precedes_new_material(self):
        self.write("reports/a.md", "new")
        self.write("reports/z.md", "old")
        old = self.git("hash-object", "--", "reports/z.md")
        self.write("docs/wiki/log.md", f"2026-11-26 start rev:{self.start}\n"
                   f"2026-11-26 ingest reports/z.md blob:{old} → z\n")
        self.write("reports/z.md", "corrected")
        for summary in ("", f"## 元の材料\n\nblob:{old}\n"):
            if summary:
                self.write("docs/wiki/summary/reports/z.md", summary)
            first = self.scan()["pending"][:1]
            self.assertEqual([(row["path"], row["state"]) for row in first],
                             [("reports/z.md", "更新あり")])


    def test_incoming_wiki_link_with_title(self):
        page = "docs/notes.md"
        self.write(page, "[a](wiki/old.md '日本語')\n[b](wiki/old.md (説明))\n")
        rows = [r for r in self.scan()["warnings"] if r["page"] == page]
        self.assertEqual([r["line"] for r in rows], [1, 2])


    def test_cross_source_claim_requires_distinct_originals(self):
        self.write("reports/a.md", "# a\n")
        self.write("reports/b.md", "# b\n")
        page = "docs/wiki/concepts/methods/probe.md"
        for other, warned in (("./a.md#two", True), ("b.md", False)):
            self.write(page, "## 横断的知見\n\n- 共通点（出典: "
                       "[一](../../../../reports/a.md#one)、"
                       f"[二](../../../../reports/{other})）\n")
            self.assertEqual(any(w["rule"] == "横断的知見の出典"
                                 for w in self.scan()["warnings"]), warned)


if __name__ == "__main__":
    unittest.main()
