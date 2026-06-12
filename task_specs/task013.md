## task013 の変換ロジック

参照データでは、各例の入力と出力は同じサイズで、入力には背景 `0` 以外の単独セルがちょうど2個あります。 

この2個の色付きセルを「周期パターンの種」として使い、グリッドの長い方向へ繰り返し展開します。

## ルール

### 1. 出力サイズ

出力は入力と同じ `H x W`。

最初はすべて `0` で初期化する。

### 2. 入力中の2つの非ゼロセルを取得する

非ゼロセルを2個見つける。

それぞれを次のように表す。

```text
(r1, c1, color1)
(r2, c2, color2)
```

### 3. グリッドが横長なら「縦線」を繰り返す

条件：

```text
W > H
```

この場合、列方向に周期を作り、各対象列を上から下まで同じ色で塗る。

手順：

1. 2つの非ゼロセルを `列 c` の小さい順に並べる。
2. 左側のセルを `(ra, ca, colorA)`、右側のセルを `(rb, cb, colorB)` とする。
3. 列間隔を `d = cb - ca` とする。
4. `ca` から右方向へ、`d` 列おきに列を選ぶ。
5. 選んだ列を、`colorA, colorB, colorA, colorB, ...` の順に交互に塗る。
6. 各選択列は全行を同じ色で塗る。

つまり、対象列は次の通り。

```text
ca, ca+d, ca+2d, ca+3d, ...
```

色は次の通り。

```text
colorA, colorB, colorA, colorB, ...
```

列がグリッド外に出たら終了する。

列 `ca` より左側はそのまま `0`。

### 4. グリッドが縦長なら「横線」を繰り返す

条件：

```text
H > W
```

この場合、行方向に周期を作り、各対象行を左から右まで同じ色で塗る。

手順：

1. 2つの非ゼロセルを `行 r` の小さい順に並べる。
2. 上側のセルを `(ra, ca, colorA)`、下側のセルを `(rb, cb, colorB)` とする。
3. 行間隔を `d = rb - ra` とする。
4. `ra` から下方向へ、`d` 行おきに行を選ぶ。
5. 選んだ行を、`colorA, colorB, colorA, colorB, ...` の順に交互に塗る。
6. 各選択行は全列を同じ色で塗る。

つまり、対象行は次の通り。

```text
ra, ra+d, ra+2d, ra+3d, ...
```

色は次の通り。

```text
colorA, colorB, colorA, colorB, ...
```

行がグリッド外に出たら終了する。

行 `ra` より上側はそのまま `0`。

## 重要な実装ポイント

* 元の色付きセルの「もう一方の座標」は無視する。

  * 横長の場合、行位置 `r` は無視し、列位置だけ使う。
  * 縦長の場合、列位置 `c` は無視し、行位置だけ使う。
* 横長の場合は、2点を列順に並べる。
* 縦長の場合は、2点を行順に並べる。
* 繰り返しは左/上には伸ばさず、右/下方向にだけ伸ばす。
* 正方形の例はない。実装上は `W >= H` を横長扱いにして縦線を生成すればよい。

## 擬似コード

```python
def solve(grid):
    H = len(grid)
    W = len(grid[0])

    points = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                points.append((r, c, grid[r][c]))

    out = [[0 for _ in range(W)] for _ in range(H)]

    if W >= H:
        # 横長: 縦線を作る
        points.sort(key=lambda x: x[1])  # column sort

        _, c0, color0 = points[0]
        _, c1, color1 = points[1]

        d = c1 - c0

        k = 0
        c = c0
        while c < W:
            color = color0 if k % 2 == 0 else color1
            for r in range(H):
                out[r][c] = color
            k += 1
            c += d

    else:
        # 縦長: 横線を作る
        points.sort(key=lambda x: x[0])  # row sort

        r0, _, color0 = points[0]
        r1, _, color1 = points[1]

        d = r1 - r0

        k = 0
        r = r0
        while r < H:
            color = color0 if k % 2 == 0 else color1
            for c in range(W):
                out[r][c] = color
            k += 1
            r += d

    return out
```

## test[0] への適用

`test[0]` は `11 x 27` なので横長。

入力の非ゼロセルは次の2つ。

```text
(0, 5) = 3
(10, 10) = 4
```

列順では `5` が先、`10` が後。

```text
ca = 5, colorA = 3
cb = 10, colorB = 4
d = 10 - 5 = 5
```

したがって、全行に対して次の列を塗る。

```text
列 5  -> 3
列 10 -> 4
列 15 -> 3
列 20 -> 4
列 25 -> 3
```

それ以外は `0`。
このため、出力は画像のように緑・黄・緑・黄・緑の縦線パターンになる。
