# DefaultDQN に TBO の公開範囲を限定する

Transformed Bellman Operator は DQN 系の Bellman ターゲットと Q 値メトリクスの意味を変えるため、anet-lab では DefaultDQN にだけ公開する。Rainbow はオリジナルのアルゴリズム構成を保つため TBO を強制 OFF とし、MuZero は独自の値変換・分布表現へ発展する余地があるため今回の対象外にする。
