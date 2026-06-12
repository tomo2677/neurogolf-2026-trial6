## task019 変換ロジック

参照データでは、各入力は `H×W`、出力は必ず `2H×2W` になっています。変換は **入力を縦横に2回ずつタイル複製し、その後、色付きセルの斜め隣に 8 を置く** ルールです。

---

## ルール

### 1. 出力サイズを入力の2倍にする

入力が `H×W` のとき、出力は

```text
2H × 2W
```

にする。

---

### 2. 入力を 2×2 にタイルする

まず、入力グリッドを縦方向・横方向に2回ずつ繰り返す。

つまり、出力座標 `(r, c)` の初期値は

```text
input[r % H][c % W]
```

になる。

例：入力内の色付きセルが `(1, 1)` にあるなら、出力では

```text
(1, 1)
(1, 1 + W)
(1 + H, 1)
(1 + H, 1 + W)
```

に同じ色が置かれる。

---

### 3. 色付きセルの「斜め隣」に 8 を置く

タイル後の出力グリッドを見て、色付きセルの斜め4方向にある黒セル `0` を `8` に変える。

斜め4方向とは以下。

```text
(r-1, c-1)
(r-1, c+1)
(r+1, c-1)
(r+1, c+1)
```

ただし、出力範囲外には何もしない。

---

### 4. 重要な優先順位

元の色付きセルは絶対に `8` で上書きしない。

つまり、

```text
すでに非ゼロ色ならそのまま
0 で、かつ斜め隣に色付きセルがあるなら 8
それ以外は 0
```

とする。

また、`8` を置いた結果をさらに使って広げることはしない。
判定に使うのは、あくまで **2×2タイル後に存在する元の色付きセル** だけ。

---

## 実装用の擬似コード

```python
def solve(grid):
    H = len(grid)
    W = len(grid[0])

    # 1. 入力を 2×2 にタイル
    out = [[grid[r % H][c % W] for c in range(2 * W)] for r in range(2 * H)]

    # 2. 8 の配置判定用に、タイル直後の色付きセル位置を保存
    colored = [[out[r][c] != 0 for c in range(2 * W)] for r in range(2 * H)]

    # 3. 黒セルのうち、斜め隣に色付きセルがあるものを 8 にする
    for r in range(2 * H):
        for c in range(2 * W):
            if out[r][c] != 0:
                continue

            has_diag_colored = False
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < 2 * H and 0 <= cc < 2 * W:
                    if colored[rr][cc]:
                        has_diag_colored = True

            if has_diag_colored:
                out[r][c] = 8

    return out
```

---

## ONNX化しやすい形での表現

テンソル演算としては以下で表せます。

```text
base[r, c] = input[r % H, c % W]

colored = base != 0

diag_mask =
    shift(colored, +1, +1) OR
    shift(colored, +1, -1) OR
    shift(colored, -1, +1) OR
    shift(colored, -1, -1)

output = where(colored, base,
               where(diag_mask, 8, 0))
```

ここで `shift` は範囲外を `False` で埋めるシフトです。
`diag_mask` は斜め隣に元の色付きセルがある場所だけ `True` になります。

---

## test[0] への適用

入力は `6×5` なので、出力は `12×10`。
まず入力を 2×2 に繰り返し、緑 `3` のセルを4か所ずつ複製します。
その後、それらの `3` の斜め隣の黒セルだけを `8` にします。
これで提示された `12×10` の出力と一致します。
