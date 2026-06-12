## task036 の変換ルール

入力は 30×30 で、背景 `0` の上に、多数の小さいノイズ点と、1つだけ大きな同色連結オブジェクトがあります。出力は、その **最大の同色連結オブジェクトだけを最小外接矩形で切り出したもの** です。提示データの入出力サイズは train[0] が 30×30→5×3、train[1] が 30×30→3×3、test[0] が 30×30→4×4 です。

## 実装すべき処理

1. 背景 `0` を無視する。
2. `0` 以外の各セルについて、**同じ色どうしが上下左右で接している 4近傍連結成分** をすべて探す。
3. 連結成分のうち、セル数が最大のものを対象オブジェクトとする。

   * 色は固定ではない。
   * train[0] では `3`、train[1] では `4`、test[0] では `2` が対象。
   * 他の色はランダムに散らばるノイズなので無視する。
4. 対象オブジェクトのセル座標から、最小行 `r_min`、最大行 `r_max`、最小列 `c_min`、最大列 `c_max` を求める。
5. その矩形範囲を出力する。

   * 対象オブジェクトのセルは元の色で出す。
   * 対象オブジェクトでない位置は `0` にする。
   * 内部の穴や欠けは `0` のまま残す。
   * 回転、反転、拡大縮小、色変更はしない。

## 例での対応

### train[0]

最大連結成分は緑 `3` のオブジェクト。
0-index 座標では、対象の外接矩形は `rows 10..14, cols 17..19`。

切り出し結果は：

```text
3 3 0
3 3 3
3 0 3
3 3 3
0 3 3
```

### train[1]

最大連結成分は黄 `4` のオブジェクト。
外接矩形は `rows 9..11, cols 11..13`。

```text
0 4 0
4 4 4
0 4 4
```

### test[0]

最大連結成分は赤 `2` のオブジェクト。
外接矩形は `rows 12..15, cols 19..22`。

したがって出力は：

```text
0 2 2 2
2 2 0 2
2 2 0 2
0 2 2 2
```

これは task036.json 内の test[0] の出力配列とも一致します。

## 擬似コード

```python
grid = input
H, W = 30, 30

visited = [[False]*W for _ in range(H)]
components = []

for r in range(H):
    for c in range(W):
        if grid[r][c] == 0 or visited[r][c]:
            continue

        color = grid[r][c]
        # BFS/DFS: same-color 4-neighbor component
        comp = []
        stack = [(r, c)]
        visited[r][c] = True

        while stack:
            x, y = stack.pop()
            comp.append((x, y))

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if not visited[nx][ny] and grid[nx][ny] == color:
                        visited[nx][ny] = True
                        stack.append((nx, ny))

        components.append((color, comp))

# 最大の同色連結成分を選ぶ
target_color, target_cells = max(components, key=lambda x: len(x[1]))
target_set = set(target_cells)

r_min = min(r for r, c in target_cells)
r_max = max(r for r, c in target_cells)
c_min = min(c for r, c in target_cells)
c_max = max(c for r, c in target_cells)

out = []
for r in range(r_min, r_max + 1):
    row = []
    for c in range(c_min, c_max + 1):
        if (r, c) in target_set:
            row.append(target_color)
        else:
            row.append(0)
    out.append(row)
```

要するに、**最大の同色4連結成分を見つけ、その形を外接矩形でトリミングするタスク** です。
