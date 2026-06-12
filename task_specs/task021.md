# task021 ロジック言語化

## 変換ルール

入力は、**背景色**の盤面に、別の1色で **縦線・横線の罫線** が引かれている形です。

出力は、入力内の罫線で区切られた **大きな区画の個数** を表す小さな長方形です。

具体的には、

* 罫線色で完全に埋まっている行を「横罫線」とみなす。
* 罫線色で完全に埋まっている列を「縦罫線」とみなす。
* 横罫線によって分割された、罫線ではない行の連続ブロック数を出力の高さにする。
* 縦罫線によって分割された、罫線ではない列の連続ブロック数を出力の幅にする。
* 出力はすべて背景色で塗りつぶす。
* 罫線色は出力には使わない。
* 各区画の実際の大きさは無視し、「何個の区画に分かれているか」だけを見る。

添付データでも、train/test の形状は 15×15→2×4、11×11→3×2、27×27→6×5、22×22→5×3 となっており、この「区画数への圧縮」と一致します。

## 例での確認

### train[0]

入力は背景色 `3`、罫線色 `7`。

横罫線は1本なので、行方向は上下の **2区画** に分かれる。
縦罫線は3本なので、列方向は **4区画** に分かれる。

したがって出力サイズは `2×4`。
色は背景色 `3` で全部埋める。

```text
出力 = 2行4列、すべて 3
```

### train[1]

入力は背景色 `1`、罫線色 `8`。

横罫線が2本あり、行方向は **3区画**。
縦罫線が1本あり、列方向は **2区画**。

したがって出力サイズは `3×2`。
色は背景色 `1`。

```text
出力 = 3行2列、すべて 1
```

### train[2]

入力は背景色 `3`、罫線色 `1`。

横罫線が5本あり、行方向は **6区画**。
縦罫線が4本あり、列方向は **5区画**。

したがって出力サイズは `6×5`。
色は背景色 `3`。

```text
出力 = 6行5列、すべて 3
```

## test[0]

入力は背景色 `1`、罫線色 `5`。

横罫線は以下の4本。

```text
row = 2, 7, 12, 17
```

これにより、行方向は次の5区画に分かれる。

```text
rows 0-1
rows 3-6
rows 8-11
rows 13-16
rows 18-21
```

縦罫線は以下の2本。

```text
col = 15, 20
```

これにより、列方向は次の3区画に分かれる。

```text
cols 0-14
cols 16-19
col 21
```

したがって、出力は `5×3`。
色は背景色 `1` で全セルを埋める。

```python
[
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1],
]
```

## 実装方針

```python
def solve(grid):
    H = len(grid)
    W = len(grid[0])

    colors = sorted(set(v for row in grid for v in row))

    # 罫線色を探す。
    # 罫線色は、少なくとも1つの「全セル同色の行」または
    # 「全セル同色の列」を作っている色。
    def full_line_count(c):
        row_count = sum(all(grid[r][x] == c for x in range(W)) for r in range(H))
        col_count = sum(all(grid[y][cidx] == c for y in range(H)) for cidx in range(W))
        return row_count + col_count

    line_color = max(colors, key=full_line_count)

    # 背景色は罫線色ではない色。
    bg_color = next(c for c in colors if c != line_color)

    # 横罫線・縦罫線を判定する。
    sep_rows = [
        all(grid[r][x] == line_color for x in range(W))
        for r in range(H)
    ]
    sep_cols = [
        all(grid[y][c] == line_color for y in range(H))
        for c in range(W)
    ]

    # 罫線ではない連続ブロック数を数える。
    def count_non_separator_blocks(flags):
        count = 0
        in_block = False
        for is_sep in flags:
            if is_sep:
                in_block = False
            else:
                if not in_block:
                    count += 1
                    in_block = True
        return count

    out_h = count_non_separator_blocks(sep_rows)
    out_w = count_non_separator_blocks(sep_cols)

    return [[bg_color for _ in range(out_w)] for _ in range(out_h)]
```

## 重要なポイント

このタスクでは、入力の罫線によってできる各長方形領域の **サイズ** は出力に反映しません。
反映するのは、縦方向に何区画、横方向に何区画あるかだけです。

つまり、入力全体を「罫線で区切られた表」と見なし、その表の各マスを1セルに縮小して、背景色だけで描き直すタスクです。
