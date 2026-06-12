## task025 変換ロジック

対象データは `task025` で、train 3件・test 1件、入出力サイズは常に同じです。

このタスクは、**全面に伸びた同色の直線を基準線として、その色と同じ小さな点だけを基準線の隣へ射影する**タスクです。

---

## ルール概要

入力には、以下の2種類の要素があります。

1. **基準線**

   * ある1色で、行全体または列全体を埋めている直線。
   * 横線の場合は「その行の全セルが同じ非ゼロ色」。
   * 縦線の場合は「その列の全セルが同じ非ゼロ色」。

2. **散らばった点**

   * 基準線と同じ色の点もある。
   * 基準線に存在しない色の点もある。
   * 基準線に存在しない色の点はノイズとして消す。

出力では、**基準線はそのまま残す**。
そして、**基準線と同じ色の点だけを、その基準線のすぐ隣に移動する**。

---

## 射影の方法

### 1. 基準線が縦線の場合

縦線の列を `C`、点の位置を `(r, c)`、色を `v` とする。

点の色 `v` が、その縦線の色と同じなら処理する。

* 点が縦線の左側にある場合、つまり `c < C`

  * 出力位置は `(r, C - 1)`
* 点が縦線の右側にある場合、つまり `c > C`

  * 出力位置は `(r, C + 1)`

つまり、**行番号はそのまま、列だけを基準線の隣に寄せる**。

例：

```text
縦線が color=3, col=11 にある
color=3 の点が row=9, col=13 にある
→ 右側から来た点なので row=9, col=12 に置く
```

---

### 2. 基準線が横線の場合

横線の行を `R`、点の位置を `(r, c)`、色を `v` とする。

点の色 `v` が、その横線の色と同じなら処理する。

* 点が横線の上側にある場合、つまり `r < R`

  * 出力位置は `(R - 1, c)`
* 点が横線の下側にある場合、つまり `r > R`

  * 出力位置は `(R + 1, c)`

つまり、**列番号はそのまま、行だけを基準線の隣に寄せる**。

例：

```text
横線が color=2, row=3 にある
color=2 の点が row=0, col=3 にある
→ 上側から来た点なので row=2, col=3 に置く
```

---

## 消えるもの

以下は出力に残さない。

* 基準線と同じ色ではない散らばった点
* 基準線を持たない色の点
* 移動前の点の元位置

出力に残るのは、

```text
基準線そのもの
+
基準線と同色の点を基準線の隣に射影したもの
```

だけです。

---

## 実装手順

0始まりインデックスで説明します。

```python
H = height
W = width
grid = input
out = 全セル0の H x W
```

### Step 1: 基準線を検出する

```python
lines = []

# 横線を検出
for r in range(H):
    color = grid[r][0]
    if color != 0 and all(grid[r][c] == color for c in range(W)):
        lines.append(("row", color, r))

# 縦線を検出
for c in range(W):
    color = grid[0][c]
    if color != 0 and all(grid[r][c] == color for r in range(H)):
        lines.append(("col", color, c))
```

### Step 2: 基準線を出力へコピーする

```python
for kind, color, idx in lines:
    if kind == "row":
        for c in range(W):
            out[idx][c] = color
    else:  # kind == "col"
        for r in range(H):
            out[r][idx] = color
```

### Step 3: 入力中の各非ゼロセルを、同色の基準線へ射影する

```python
for r in range(H):
    for c in range(W):
        v = grid[r][c]
        if v == 0:
            continue

        for kind, color, idx in lines:
            if v != color:
                continue

            if kind == "row":
                R = idx

                # 基準線上のセルは線そのものなので移動対象外
                if r == R:
                    continue

                if r < R:
                    rr = R - 1
                else:
                    rr = R + 1

                if 0 <= rr < H:
                    out[rr][c] = v

            else:  # kind == "col"
                C = idx

                # 基準線上のセルは線そのものなので移動対象外
                if c == C:
                    continue

                if c < C:
                    cc = C - 1
                else:
                    cc = C + 1

                if 0 <= cc < W:
                    out[r][cc] = v
```

---

## test[0] への適用

test[0] では縦線が3本あります。

```text
color=2 の縦線: col=4
color=3 の縦線: col=11
color=4 の縦線: col=20
```

そのため、同じ色の点だけを、それぞれの縦線の左右隣へ寄せます。

例：

```text
color=2 の点: row=2, col=16
縦線 color=2 は col=4
点は右側にある
→ row=2, col=5 に置く

color=2 の点: row=10, col=1
縦線 color=2 は col=4
点は左側にある
→ row=10, col=3 に置く

color=3 の点: row=9, col=13
縦線 color=3 は col=11
点は右側にある
→ row=9, col=12 に置く

color=4 の点: row=7, col=6
縦線 color=4 は col=20
点は左側にある
→ row=7, col=19 に置く
```

色 `8` の点は、色 `8` の基準線が存在しないためすべて消えます。

---

## まとめ

このタスクは、次の一文で表せます。

**全行または全列を占める同色直線を基準線として残し、基準線と同じ色の散在セルだけを、その基準線のすぐ隣へ垂直方向または水平方向に投影する。基準線を持たない色の散在セルは削除する。**
