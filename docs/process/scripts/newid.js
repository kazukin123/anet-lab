#!/usr/bin/env node
/**
 * 記録の ID を発番する。
 *
 *   node <このファイルのパス> <接頭辞> [件数]
 *
 * 接頭辞は `issue` か `worklog` を渡す。末尾のハイフンは付けない。
 * 件数を省略すると 1 件出す。出力は拡張子を含まない ID である。
 *
 * シェルに依存しない。POSIX シェルでも PowerShell でも cmd でも同じ 1 行で動く。
 * 呼び出し側に引用符が要らないため、シェルごとのエスケープの違いに当たらない。
 *
 * 乱数は crypto.randomBytes（暗号論的な乱数生成器）を使う。
 * AI に値を出させてはならない。大規模言語モデルの乱数生成は、1 回に 1 個を
 * 独立した呼び出しで出すとき最も崩れることが分かっている
 * （Large Language Models Are Bad Dice Players, arXiv:2601.05414）。
 *
 * 既存のファイルと照らして重なりを避けることもできない。照らせるのは今のブランチだけであり、
 * ブランチをまたぐ一意性は乱数の質そのものに依存する。
 */

const crypto = require('crypto');

const prefix = process.argv[2];
const count = process.argv[3] === undefined ? 1 : Number(process.argv[3]);

if (!prefix || !/^[a-z][a-z0-9]*$/.test(prefix)) {
  console.error('接頭辞を指定してください（例: issue / worklog）。英小文字と数字のみ、ハイフンは含めません');
  process.exit(1);
}

if (!Number.isInteger(count) || count < 1 || count > 100) {
  console.error('件数は 1 から 100 の整数で指定してください');
  process.exit(1);
}

const d = new Date();
const z = (n) => String(n).padStart(2, '0');
const stamp = d.getFullYear() + z(d.getMonth() + 1) + z(d.getDate());

for (let i = 0; i < count; i += 1) {
  console.log(prefix + '-' + stamp + '-' + crypto.randomBytes(3).toString('hex'));
}
