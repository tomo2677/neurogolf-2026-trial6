## タスク29の変換ロジック

入力グリッド内に、**1色だけで作られた長方形の枠**が1つ存在します。出力は、その枠線を取り除いた **内側だけをそのまま切り出した矩形**です。train/test の入出力サイズもこの「枠の内側サイズ」と一致しています。

## 観察された対応

* train[0]
  色 `2` が長方形の枠を作っている。
  枠の外接範囲は `row=3..10, col=4..13`。
  出力は枠を除いた `row=4..9, col=5..12` の `6x8`。

* train[1]
  色 `4` が長方形の枠を作っている。
  枠の外接範囲は `row=6..12, col=2..6`。
  出力は枠を除いた `row=7..11, col=3..5` の `5x3`。

* train[2]
  色 `8` が長方形の枠を作っている。
  枠の外接範囲は `row=2..11, col=3..14`。
  出力は枠を除いた `row=3..10, col=4..13` の `8x10`。

* test[0]
  色 `3` が長方形の枠を作っている。
  枠の外接範囲は `row=4..13, col=3..13`。
  出力は枠を除いた `row=5..12, col=4..12` の `8x9`。

## 実装方針

1. 入力グリッドに存在する各色 `c` について、その色の全座標を集める。
2. その座標群の外接矩形を求める。
   `r_min, r_max, c_min, c_max`
3. その外接矩形が、色 `c` の**完全な長方形枠**になっているか判定する。

   * 上辺: `grid[r_min][c_min..c_max]` がすべて `c`
   * 下辺: `grid[r_max][c_min..c_max]` がすべて `c`
   * 左辺: `grid[r_min..r_max][c_min]` がすべて `c`
   * 右辺: `grid[r_min..r_max][c_max]` がすべて `c`
   * かつ、色 `c` のセル数が枠線セル数
     `2 * ((r_max-r_min+1) + (c_max-c_min+1)) - 4`
     と一致する。
4. 条件を満たす色が枠色。
5. 出力は以下の範囲をそのまま切り出す。

```python
output = grid[r_min+1 : r_max, c_min+1 : c_max]
```

## 擬似コード

```python
def solve(grid):
    H = len(grid)
    W = len(grid[0])

    colors = sorted(set(v for row in grid for v in row))

    for color in colors:
        coords = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color:
                    coords.append((r, c))

        if not coords:
            continue

        r0 = min(r for r, c in coords)
        r1 = max(r for r, c in coords)
        c0 = min(c for r, c in coords)
        c1 = max(c for r, c in coords)

        # 内側が存在する必要がある
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        expected = set()

        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))

        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))

        actual = set(coords)

        if actual == expected:
            return [
                row[c0 + 1 : c1]
                for row in grid[r0 + 1 : r1]
            ]
```

## テストケースへの適用結果

test[0] では、色 `3` の枠の内側を取り出すため、出力は次の通りです。

```text
[
  [2, 0, 8, 1, 1, 1, 0, 1, 0],
  [8, 1, 0, 8, 2, 8, 1, 2, 8],
  [8, 2, 0, 2, 0, 1, 1, 8, 1],
  [0, 1, 8, 8, 1, 1, 8, 1, 8],
  [0, 1, 8, 8, 0, 8, 0, 2, 0],
  [0, 8, 8, 2, 8, 8, 8, 8, 8],
  [8, 0, 2, 0, 0, 0, 0, 8, 8],
  [0, 2, 8, 8, 1, 2, 0, 0, 2]
]
```

このタスクは、色の変換・回転・反転・縮小ではなく、**単色の長方形枠を検出して、その内側領域をクロップするだけ**の処理です。
