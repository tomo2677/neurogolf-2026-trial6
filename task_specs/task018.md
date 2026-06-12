# task018 変換ロジック

このタスクは、**完成済みの小さな図形テンプレートを見つけ、別の場所に置かれた3つの目印セルに合わせて、同じ図形を回転・反転して復元する**タスクです。

## 1. 入力に含まれる要素

入力グリッドには次の2種類の非ゼロ要素があります。

### A. 完成済みテンプレート

4近傍で連結している非ゼロ成分のうち、**同じ色が複数回出てくる成分**をテンプレートとみなします。

テンプレート内では、

* 複数回出てくる色が「本体色」
* 本体色以外の3セルが「目印色」
* 目印色はそれぞれ1回ずつ出現する

という構造になっています。

例として train[2] では、水色 `8` が本体色で、`1,2,4` が目印色です。

### B. 目印だけの配置先

テンプレートから離れた場所に、テンプレート内の目印色3個だけが置かれています。

この3個の目印セルは、完成済みテンプレートをどの位置・どの向きに置くべきかを指定しています。

目印セル同士は、隣接している場合も離れている場合もあります。
そのため、**目印セットは連結成分として探すのではなく、テンプレートの3つの目印色の相対配置に一致するかどうかで探します。**

## 2. 出力で行うこと

出力は入力と同じサイズです。

最初に全セルを `0` にします。

そのうえで、各テンプレートについて、入力中にある「目印だけの配置先」を探し、そこにテンプレート全体を描画します。

重要なのは、**元のテンプレート自体は出力に残さない**ことです。
出力には、配置先に復元されたコピーだけが残ります。

## 3. 向きの決め方

テンプレートは、配置先の3つの目印セルに合うように、次の8通りの向きのどれかで置かれます。

* そのまま
* 上下反転
* 左右反転
* 180度回転
* 転置
* 反転転置
* 90度回転
* 270度回転

実装上は、座標 `(r, c)` に対して次の8変換を試せばよいです。

```text
(r,  c)
(r, -c)
(-r, c)
(-r,-c)
(c,  r)
(c, -r)
(-c, r)
(-c,-r)
```

その後に平行移動を加えることで、配置先の目印セルに合わせます。

## 4. 厳密な判定条件

各テンプレートについて、8通りの変換と平行移動を全探索します。

ある変換・平行移動が有効になる条件は次の通りです。

1. テンプレート内の3つの目印セルを変換・平行移動した位置が、入力グリッド上に存在する。
2. その3位置には、テンプレート内と同じ目印色が置かれている。
3. その3位置は、元のテンプレート成分の一部ではない。
4. テンプレート内の本体色セルを変換・平行移動した位置は、入力ではすべて `0` である。
5. テンプレート全体がグリッド範囲内に収まる。

この条件を満たした場合、その位置にテンプレート全体を出力へ描画します。

## 5. 擬似コード

```python
def solve(grid):
    H, W = len(grid), len(grid[0])
    output = [[0 for _ in range(W)] for _ in range(H)]

    # 1. 非ゼロセルの4近傍連結成分を取得
    components = find_4_connected_nonzero_components(grid)

    sources = []
    source_mask = set()

    # 2. テンプレート成分を抽出
    for comp in components:
        # comp: [(r, c, color), ...]
        color_count = Counter(color for r, c, color in comp)

        # 同じ色が複数回出る成分をテンプレートとする
        if max(color_count.values()) >= 2:
            base_color = argmax_count(color_count)

            cells = comp
            markers = [
                (r, c, color)
                for r, c, color in comp
                if color != base_color
            ]

            # このタスクでは markers は常に3個
            sources.append({
                "cells": cells,
                "base": base_color,
                "markers": markers,
            })

            for r, c, color in comp:
                source_mask.add((r, c))

    # 3. テンプレート以外の非ゼロセルを、配置先候補として色別に持つ
    target_by_color = defaultdict(list)

    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            if color != 0 and (r, c) not in source_mask:
                target_by_color[color].append((r, c))

    # 4. 8通りの座標変換
    transforms = [
        lambda r, c: ( r,  c),
        lambda r, c: ( r, -c),
        lambda r, c: (-r,  c),
        lambda r, c: (-r, -c),
        lambda r, c: ( c,  r),
        lambda r, c: ( c, -r),
        lambda r, c: (-c,  r),
        lambda r, c: (-c, -r),
    ]

    # 5. 各テンプレートを、目印3点に合う場所へコピー
    for source in sources:
        cells = source["cells"]
        base = source["base"]
        markers = source["markers"]

        # 探索開始用の目印を1つ選ぶ
        # 候補数が少ない色を選ぶと効率がよい
        anchor = marker_with_fewest_targets(markers, target_by_color)
        ar, ac, acolor = anchor

        for transform in transforms:
            tar, tac = transform(ar, ac)

            for rr, cc in target_by_color[acolor]:
                # anchor が (rr, cc) に来るような平行移動
                dr = rr - tar
                dc = cc - tac

                placed_cells = []
                ok = True

                # テンプレート全体が範囲内か確認
                for r, c, color in cells:
                    tr, tc = transform(r, c)
                    nr = tr + dr
                    nc = tc + dc

                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break

                    placed_cells.append((nr, nc, color))

                if not ok:
                    continue

                # 目印3個が入力上の配置先セルと一致するか確認
                for r, c, color in markers:
                    tr, tc = transform(r, c)
                    nr = tr + dr
                    nc = tc + dc

                    if grid[nr][nc] != color:
                        ok = False
                        break

                    # 元テンプレート自身を配置先として使わない
                    if (nr, nc) in source_mask:
                        ok = False
                        break

                if not ok:
                    continue

                # 本体色セルの位置は、入力では空白でなければならない
                for nr, nc, color in placed_cells:
                    if color == base and grid[nr][nc] != 0:
                        ok = False
                        break

                if not ok:
                    continue

                # 条件を満たしたので、テンプレート全体を出力に描画
                for nr, nc, color in placed_cells:
                    output[nr][nc] = color

    return output
```

## 6. train/test での見え方

train[0] では、テンプレートが2個あります。
それぞれのテンプレートに対して、離れた位置にある3つの目印セルを見つけ、目印の相対配置に合うように回転してコピーしています。

train[1] では、緑 `3` を本体色とする縦長テンプレートを、右下の `1,2,4` の目印配置に合わせて横向きに回転して復元しています。

train[2] では、水色 `8` のテンプレートを、下側の `1,2,4` の目印配置に合わせて上下反転して復元しています。

test[0] では、灰色 `5` を本体色とするテンプレートが2個あります。
それぞれについて、下側に置かれている `1,2,4` の目印3点に合うように、片方は回転、もう片方は転置方向でコピーされています。元の上側テンプレートは出力から消え、下側の復元結果だけが残ります。
