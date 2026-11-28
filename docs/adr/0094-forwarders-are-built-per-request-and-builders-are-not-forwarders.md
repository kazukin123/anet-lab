# Forwarder は依頼ごとに作り、作る側（並びと部品）は Forwarder を兼ねない

実行の口が 2 つ（`Forwarder` と、開始の印を受ける実行器）あり、並びは Forwarder と作る側を兼ね、依頼の種類は RL に依らない層の列挙だった（[RFD0200](../memo/0200_forwarder_public_interface_rework_10rfd.md)。[ADR 0084](0084-forwarder-is-common-interface-and-chain-serves-requests-named-by-input-and-product.md) の一部を改める）。**実行するものを `Forwarder` の 1 つにして、行ごとの履歴を使う変換も依頼で作った Forwarder が受け（開始の印は入力の鍵にする）、並び（`ForwarderBuilder`）と部品は「依頼を受けて Forwarder を作るもの」に限り、依頼は、RL の側が定義する種類の名前と付帯の情報で運ぶ**ことを決定する。使う側の使い方が「依頼を出して、受け取った Forwarder を呼ぶ」の 1 つになり、RL に依らない層が、依頼の種類を解釈しなくて済む。

## Considered Options

- **実行器を残し、開始の印を型つきの引数で渡す**: 口が 2 つのままで、RL の概念が汎用の口の引数に残る。Network の実行器は、開始の印を呼び出しごとの型つきの値で受けるので、形はそろわなくなるが、`Forwarder` の口は入力と capture だけにする。却下。
- **並びが Forwarder を兼ねる**: 依頼なしで spec を答える口が残り、どの依頼の spec かが決まらない。却下。
- **依頼の種類を、汎用の層の列挙のままにする**: 種類を足すたびに、汎用の層を変えることになる。却下。
