# task014 のロジック

タスク14は、**0（黒）で区切られた 2×2 の4つの矩形ブロックから、「色が1つだけ異なるブロック」をそのまま抜き出す**タスクです。入力には train 3件 / test 1件があり、各入出力サイズもこのブロック抽出結果と一致しています。

## 観察される構造

入力グリッドは、必ず以下のような構造です。

* 背景色は `0`
* 横方向に、全セル `0` の行の帯で上下に分割される
* 縦方向に、全セル `0` の列の帯で左右に分割される
* 結果として、非ゼロの矩形領域が **2行×2列 = 4ブロック** できる
* 各ブロックは、`0` と 1種類の非ゼロ色だけで構成されている
* 4ブロックのうち、3ブロックは同じ色、1ブロックだけ別の色
* 出力は、その **1ブロックだけに現れる色のブロック** を、内部の `0` も含めて矩形のまま切り出したもの

## 例での対応

train[0] では、4ブロックの色は以下です。

* 左上: `8`
* 右上: `8`
* 左下: `2`
* 右下: `8`

`2` のブロックだけが1回しか出てこないので、左下ブロックをそのまま出力します。

train[1] では `2` が3ブロック、`3` が1ブロックなので、`3` の右下ブロックを出力します。

train[2] では `1` が3ブロック、`4` が1ブロックなので、`4` の左下ブロックを出力します。

test[0] では `3` が3ブロック、`1` が1ブロックなので、左上の `1` ブロックを出力します。test[0] の出力は `6×6` です。

```python
[
    [1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
]
```

## 実装手順

1. 入力グリッド `grid` に対して、各行が非ゼロを含むかを調べる。

   * `row_has_color[r] = any(grid[r][c] != 0)`

2. `row_has_color` の連続した `True` 区間を取る。

   * これが上段ブロック群、下段ブロック群になる。
   * 例: `[(0, 6), (10, 15)]`

3. 同様に、各列が非ゼロを含むかを調べる。

   * `col_has_color[c] = any(grid[r][c] != 0)`

4. `col_has_color` の連続した `True` 区間を取る。

   * これが左ブロック群、右ブロック群になる。
   * 例: `[(0, 6), (8, 17)]`

5. 行区間 × 列区間の組み合わせで、4つの矩形ブロックを得る。

6. 各ブロック内に含まれる非ゼロ色を調べる。

   * 各ブロックには非ゼロ色が1種類だけある。
   * その色をそのブロックの代表色とする。

7. 代表色をブロック単位で数える。

   * セル数ではなく、**ブロック数**で数えること。
   * 3回出る色と、1回だけ出る色がある。

8. 1回だけ出る色を持つブロックを選ぶ。

9. そのブロックの矩形範囲を、`0` も含めてそのまま返す。

## 擬似コード

```python
def runs(mask):
    result = []
    i = 0
    n = len(mask)

    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i
        while i < n and mask[i]:
            i += 1
        end = i

        result.append((start, end))

    return result


def solve(grid):
    H = len(grid)
    W = len(grid[0])

    # 1. 非ゼロを含む行・列を検出
    row_has = []
    for r in range(H):
        row_has.append(any(grid[r][c] != 0 for c in range(W)))

    col_has = []
    for c in range(W):
        col_has.append(any(grid[r][c] != 0 for r in range(H)))

    # 2. 0だけの帯で区切られた行区間・列区間
    row_bands = runs(row_has)
    col_bands = runs(col_has)

    # 3. 4ブロックを取得し、それぞれの代表色を調べる
    blocks = []

    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            colors = set()

            for r in range(r0, r1):
                for c in range(c0, c1):
                    v = grid[r][c]
                    if v != 0:
                        colors.add(v)

            if len(colors) == 1:
                color = next(iter(colors))
                blocks.append((color, r0, r1, c0, c1))

    # 4. 色ごとのブロック数を数える
    count = {}
    for color, r0, r1, c0, c1 in blocks:
        count[color] = count.get(color, 0) + 1

    # 5. 1ブロックにしか現れない色を探す
    target_color = None
    for color, n in count.items():
        if n == 1:
            target_color = color
            break

    # 6. その色のブロックを切り出して返す
    for color, r0, r1, c0, c1 in blocks:
        if color == target_color:
            return [row[c0:c1] for row in grid[r0:r1]]
```

## 実装上の注意

重要なのは、**代表色をセル数で多数決しない**ことです。ブロックのサイズが異なる場合があるため、色の出現セル数ではなく、4つの矩形ブロックのうち何ブロックに現れるかで判定します。

また、出力時には選ばれたブロック内の `0` を削除してはいけません。`0` は模様の一部なので、選ばれた矩形範囲をそのまま返します。
