# クラスの宣言は、種類の切れ目でアクセス指定を書き直して見出しにし、アクセス指定の前に空行を入れない

## 元の材料

[ADR 0076](../../../adr/0076-class-declarations-repeat-access-specifiers-as-kind-headings-without-blank-lines.md) blob:8879e87ba2583b9cbf8d79211ce051e98e9b5029

## 要点

- 新規 C++ クラスでは型、定数、関数、データメンバの種類ごとに宣言をまとめる。
- 可視性が同じでも public や private を書き直して種類の見出しにする。
- アクセス指定の前後には空行を置かない。
- 同じ指定内の意味の異なる群は空行で分けるが、3宣言以下なら空けない。
- 既存クラスの一括修正は別課題で扱う。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
