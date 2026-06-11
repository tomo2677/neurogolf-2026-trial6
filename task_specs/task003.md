タスク3の変換ロジックはかなり単純で、**縦方向の周期パターンを9行まで延長し、色1を色2に置き換える**処理です。入力は6×3、出力は9×3です。

## 変換ルール

入力の各行を「3マス幅の行パターン」として見る。

まず、6行の並びから**最短の縦周期**を見つける。
その周期をそのまま下方向に繰り返して、出力を9行にする。

その後、色を変換する。

```text
0 → 0 のまま
1 → 2 に置換
```

つまり、形はそのまま保持し、青いセルだけ赤に変わる。

## 周期の見つけ方

入力の行列を `input[0] ... input[5]` とする。

周期 `p` を小さい順に探し、

```text
input[i] == input[i - p]
```

が、`i = p ... 5` の範囲ですべて成立する最小の `p` を採用する。

その後、出力の各行 `r` は、

```text
output[r] = input[r % p]
```

を元に作る。
ただし、セル値は `1 → 2` に置換する。

## train例での確認

### train[0]

入力の行パターンは、

```text
A B A C A B
```

これは周期4の

```text
A B A C
```

の途中までが見えている状態。

9行に延長すると、

```text
A B A C A B A C A
```

になる。
その後、1を2に変換する。

### train[1]

入力は、

```text
A B A B A B
```

なので周期2。

9行に延長すると、

```text
A B A B A B A B A
```

になる。

### train[2]

入力は、

```text
A B A A B A
```

なので周期3。

9行に延長すると、

```text
A B A A B A A B A
```

になる。

## test[0]への適用

test入力は、

```text
111
010
010
111
010
010
```

行パターンで見ると、

```text
A B B A B B
```

なので周期3。

これを9行に延長すると、

```text
A B B A B B A B B
```

になる。

さらに `1 → 2` に変換するので、出力は、

```text
222
020
020
222
020
020
222
020
020
```

になる。

## 実装向けの簡潔な表現

```python
def solve(x):
    h, w = len(x), len(x[0])

    # 最短の縦周期を探す
    p = h
    for cand in range(1, h + 1):
        ok = True
        for i in range(cand, h):
            if x[i] != x[i - cand]:
                ok = False
                break
        if ok:
            p = cand
            break

    # 周期を9行まで延長し、1を2に変換
    out = []
    for r in range(9):
        row = []
        src = x[r % p]
        for v in src:
            row.append(2 if v == 1 else v)
        out.append(row)

    return out
```

要約すると、**「6×3の入力から縦方向の最短周期を検出し、その周期を9行まで繰り返す。最後に色1を色2へ置換する」**タスクです。
