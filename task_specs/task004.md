タスク4の変換は、**各色の斜めに傾いた輪郭オブジェクトを、下辺を固定したまま1マス右へ押し込む**処理です。グリッドサイズ、背景0、色は変わりません。train/test の入出力でも、同じ「下辺固定＋右方向へ1マス移動」の規則になっています。

## 変換ルール

各非0色の連結成分を1つのオブジェクトとして扱います。斜め線でつながっているので、連結判定は **8近傍** が安全です。

各オブジェクトについて、

* そのオブジェクトの一番下の行を `bottom_row`
* そのオブジェクトの一番右の列を `right_col`

とします。

そして、オブジェクト内の各色セル `(r, c)` を次のように移動します。

```text
もし r == bottom_row:
    そのセルは動かさない
それ以外:
    1列右へ移動する
    ただし right_col を超えないようにする
```

つまり、0始まり座標で書くと、

```text
new_r = r

if r == bottom_row:
    new_c = c
else:
    new_c = min(c + 1, right_col)
```

です。

## 見た目で言うと

入力の図形は、上辺が左寄り、下辺が右寄りになっている斜めの輪郭です。出力では、その図形の**下辺はそのまま残し**、それより上の色セルだけを**右に1マスずらします**。ただし、右端にはみ出すセルは、元のオブジェクトの右端列で止めます。

例えば test の黄色オブジェクトでは、上辺は1マス右に動いていますが、最下段の黄色の横線は入力と同じ位置に残っています。右端の縦に見える部分も、右端列を超えないように詰まっています。

## 実装向けの要約

```python
output = all zeros, same shape as input

for each 8-connected nonzero component:
    color = component color
    bottom_row = max(r for r, c in component)
    right_col = max(c for r, c in component)

    for r, c in component:
        if r == bottom_row:
            nr, nc = r, c
        else:
            nr, nc = r, min(c + 1, right_col)

        output[nr][nc] = color
```

この処理を全オブジェクトに独立に適用すれば、train[0] のピンク・赤、train[1] の水色、test[0] の黄色すべてを説明できます。
