タスク9の変換は、**同じ色の2×2ブロック同士を、同じ行または同じ列にある場合だけ直線でつなぐ**処理です。
入力・出力サイズは変わらず、格子線の色と位置もそのままです。対象データは `task009` で、train は3例、test は1例です。

## ルールの言語化

盤面は、1ピクセル幅の格子線で区切られた表になっています。
各マスの中身は **2×2** の領域で、マス同士の間に格子線が1列または1行入ります。

例えば、実ピクセル上では次のような周期です。

```text
マス行: 0-1, 3-4, 6-7, 9-10, ...
格子線: 2, 5, 8, 11, ...
```

列方向も同じです。

入力には、格子線の色とは別の色で塗られた 2×2 ブロックがあります。
出力では、**同じ色のブロックが同じ論理行に2個以上あれば、その間のマスをすべてその色で埋めます。**
また、**同じ色のブロックが同じ論理列に2個以上あれば、その間のマスをすべてその色で埋めます。**

重要なのは、**最初から入力に存在している同色ブロック同士だけを見る**ことです。
一度塗り足したマスを新しい端点として、さらに別方向へ連鎖的につなぐことはしません。

## 具体例での説明

### train[0]

赤 `2` は、同じ行に2つあります。

```text
行1: col1 と col5
```

なので、その行の col1〜col5 を赤で埋めます。

さらに赤 `2` は、同じ列にも2つあります。

```text
列5: row1 と row3
```

なので、その列の row1〜row3 を赤で埋めます。

緑 `3` は同じ行に2つあるので、その間だけ横に埋めます。
青 `1` は1つしかないので変化しません。

### train[1]

赤 `2` は同じ列にあるので縦に埋めます。
えんじ `9` は同じ行にあるので横に埋めます。
水色 `8` は、同じ列にも同じ行にも複数あるので、それぞれ縦・横に埋めます。

### train[2]

緑 `3` は、同じ列にある2点を縦につなぎ、さらに同じ行にある2点を横につなぎます。
赤 `2` は同じ行にあるブロックだけを横につなぎます。

この例で重要なのは、赤 `2` の上側のブロックと、横に埋められて新しくできた赤マスを使って、追加で縦につなぐことはしない点です。
つまり、**判定に使う端点は入力時点のブロックだけ**です。

## 実装向けの手順

1. 出力を入力のコピーで初期化する。
2. 格子線の色を求める。基本的には、0以外で最も多い色が格子線色。
3. 実ピクセル座標を論理マス座標に変換する。

   * 論理マス `(r, c)` の左上は実座標 `(3*r, 3*c)`。
   * そのマス本体は `rows = 3*r, 3*r+1`、`cols = 3*c, 3*c+1` の2×2。
4. 入力から、格子線色でも0でもない2×2ブロックを抽出する。
5. 色ごとに、元から存在するブロック位置を集める。
6. 各色について、

   * 同じ論理行に2個以上ある場合、その行の最小列〜最大列をその色で塗る。
   * 同じ論理列に2個以上ある場合、その列の最小行〜最大行をその色で塗る。
7. 塗るときは、論理マスの2×2部分だけを塗る。格子線は上書きしない。

## 擬似コード

```python
out = copy(input)

grid_color = most_frequent_nonzero_color(input)

# 色ごとの元ブロック位置を集める
positions_by_color = {}

for logical_r in range(num_cell_rows):
    for logical_c in range(num_cell_cols):
        rr = 3 * logical_r
        cc = 3 * logical_c

        block = input[rr:rr+2, cc:cc+2]

        if block is uniformly one color:
            color = that color
            if color != 0 and color != grid_color:
                positions_by_color[color].append((logical_r, logical_c))

# 元の位置だけを使って補完する
for color, positions in positions_by_color.items():

    # 横方向
    for each logical row r:
        cols = [c for (rr, c) in positions if rr == r]
        if len(cols) >= 2:
            for c in range(min(cols), max(cols) + 1):
                paint logical cell (r, c) with color

    # 縦方向
    for each logical column c:
        rows = [r for (r, cc) in positions if cc == c]
        if len(rows) >= 2:
            for r in range(min(rows), max(rows) + 1):
                paint logical cell (r, c) with color
```

## test[0] に対する説明

test[0] では、格子線色は `4` です。

入力上の色ブロックは次のように見ます。

```text
8: (row1, col2), (row4, col2)
2: (row1, col5), (row6, col1), (row6, col5)
3: (row3, col7)
```

したがって、

```text
8 は同じ列 col2 に2個ある
→ row1〜row4 の col2 を 8 で縦に埋める

2 は同じ列 col5 に2個ある
→ row1〜row6 の col5 を 2 で縦に埋める

2 は同じ行 row6 に2個ある
→ col1〜col5 を 2 で横に埋める

3 は1個だけ
→ そのまま
```

これが提示されている test 出力になります。
