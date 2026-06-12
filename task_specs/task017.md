## task017 の変換ルール

task017 は、21×21 グリッド上の周期模様から、`0` で隠されたセルを復元するタスクです。出力サイズは入力と同じ 21×21 で、入力中の非 `0` セルは正しい模様の一部、`0` セルは欠損セルとして扱います。

## 核となるルール

グリッド全体は、ある正方形の基本タイルを、行方向・列方向に同じ周期 `p` で繰り返した模様です。

各セルの色は次で決まります。

```text
output[r][c] = tile[r mod p][c mod p]
```

ここで `r, c` は 0-indexed の行・列です。

重要なのは、`p` は 21 の約数である必要はないことです。
実例では次のように周期が変わります。

```text
train[0] : p = 6
train[1] : p = 7
train[2] : p = 4
test[0]  : p = 9
```

黒い `0` 領域の形そのものには意味はありません。連結成分を見たり、図形を移動したりするタスクではなく、周期模様の欠損補完です。

## 周期 `p` の検出方法

候補周期 `p = 1..21` を小さい順に試します。

ある `p` が正しい候補である条件は次の 2 つです。

1. 入力の非 `0` セルを `(r mod p, c mod p)` ごとに分類したとき、同じ分類に入る色がすべて一致する。
2. `p × p` の全てのタイル位置について、少なくとも 1 つは非 `0` の観測値がある。

この条件を満たす最小の `p` を採用します。

左上の `p×p` をそのままタイルとして使うのは危険です。左上部分にも `0` が混ざることがあるため、必ずグリッド全体から `(r mod p, c mod p)` ごとに色を集めてタイルを復元してください。

## 実装仕様

```python
def solve(grid):
    H, W = 21, 21

    for p in range(1, 22):
        tile = [[-1 for _ in range(p)] for _ in range(p)]
        ok = True

        for r in range(H):
            for c in range(W):
                v = grid[r][c]
                if v == 0:
                    continue

                rr = r % p
                cc = c % p

                if tile[rr][cc] == -1:
                    tile[rr][cc] = v
                elif tile[rr][cc] != v:
                    ok = False
                    break
            if not ok:
                break

        if not ok:
            continue

        # p×p の全タイル位置が観測済みか確認
        complete = True
        for rr in range(p):
            for cc in range(p):
                if tile[rr][cc] == -1:
                    complete = False

        if complete:
            break

    out = [[0 for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            out[r][c] = tile[r % p][c % p]

    return out
```

## test[0] で復元される基本タイル

test[0] では周期 `p = 9` です。基本タイルは次の 9×9 です。

```text
9 6 5 6 9 5 3 3 5
6 3 2 3 6 2 9 9 2
5 2 1 2 5 1 8 8 1
6 3 2 3 6 2 9 9 2
9 6 5 6 9 5 3 3 5
5 2 1 2 5 1 8 8 1
3 9 8 9 3 8 6 6 8
3 9 8 9 3 8 6 6 8
5 2 1 2 5 1 8 8 1
```

このタイルを `r mod 9, c mod 9` で 21×21 全体に繰り返したものが出力です。
