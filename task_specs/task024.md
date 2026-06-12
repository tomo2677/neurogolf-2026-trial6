# task024 の変換ロジック

対象データは `task024.json` の train/test です。

## ルール概要

入力と同じサイズのグリッドを出力する。背景は `0` のまま。
入力にある非ゼロの単点を、色ごとに決まった方向へ直線として拡張する。

色ごとの意味は次の通り。

| 色   | 変換                             |
| --- | ------------------------------ |
| `1` | その `1` が存在する **行全体** を `1` で塗る |
| `3` | その `3` が存在する **行全体** を `3` で塗る |
| `2` | その `2` が存在する **列全体** を `2` で塗る |

交差が起きる場合は、**横線である `1` / `3` の行が、縦線である `2` の列より優先**される。

つまり、各マス `(r, c)` の出力値は次の優先順位で決まる。

```text
if 入力の r 行に 1 が存在する:
    output[r][c] = 1
elif 入力の r 行に 3 が存在する:
    output[r][c] = 3
elif 入力の c 列に 2 が存在する:
    output[r][c] = 2
else:
    output[r][c] = 0
```

このタスクでは、同じ行に `1` と `3` が同時に出るケースはない想定でよい。
安全側に倒すなら、上の擬似コードの通り `1` を `3` より先に判定する。

## test[0] での確認

入力では、

```text
3 がある行: 0, 3
1 がある行: 7, 9
2 がある列: 4, 9
```

なので出力は、

```text
行 0, 3 をすべて 3
行 7, 9 をすべて 1
列 4, 9 を 2
ただし行 0, 3, 7, 9 との交差部分は横線が優先
```

となる。

## 実装向け NumPy 風疑似コード

```python
def solve(grid):
    import numpy as np

    x = np.array(grid)
    h, w = x.shape

    out = np.zeros_like(x)

    # まず 2 のある列を縦線として塗る
    cols_2 = (x == 2).any(axis=0)
    out[:, cols_2] = 2

    # 次に 3 のある行を横線として塗る
    rows_3 = (x == 3).any(axis=1)
    out[rows_3, :] = 3

    # 最後に 1 のある行を横線として塗る
    # 1 と 3 が同じ行にあるケースへの保険として 1 を最優先にする
    rows_1 = (x == 1).any(axis=1)
    out[rows_1, :] = 1

    return out.tolist()
```

## ONNX 化しやすい式

動的サイズ対応を意識するなら、ブロードキャストと `ReduceMax` / `Where` だけで書ける。

```text
mask_col2 = ReduceMax(x == 2, axis=0, keepdims=True)   # shape: 1 x W
base     = Where(mask_col2, 2, 0)                      # H x W に broadcast

mask_row1 = ReduceMax(x == 1, axis=1, keepdims=True)   # shape: H x 1
mask_row3 = ReduceMax(x == 3, axis=1, keepdims=True)   # shape: H x 1

row_value = Where(mask_row1, 1,
            Where(mask_row3, 3, 0))                    # H x 1

output = Where(row_value != 0, row_value, base)         # row_value も H x W に broadcast
```

要するに、`2` は縦方向、`1` と `3` は横方向に全延長し、交差では横方向を優先するタスク。
