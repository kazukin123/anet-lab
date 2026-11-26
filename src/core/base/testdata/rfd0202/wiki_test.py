"""RFD0202 の wiki 点検を、隔離した作業ツリーで公開 CLI から検証する。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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
            out.write("2026-11-26 query 質問 → docs/wiki/index.md\n")
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
