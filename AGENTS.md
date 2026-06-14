常に日本語で応答してください。ただし、code, command, file path, technical term は English のまま残してください。

この repo は NeuroGolf 2026 の local research / local validation / score-up 用です。

## Competition Objective

この repo の目的は、NeuroGolf 2026 で上位入賞することです。

- competition は `400` task の合計点で競います。
- functionally correct な task は `max(1, 25 - ln(cost))` で採点されます。
- 1 task の理論最大は `25` 点、400 task 合計の理論最大は `10000` 点です。
- 上位入賞には概ね `8000` 点級が必要だと考え、これを repo-internal strategic target として扱います。
- local research、validation、skill、tool、data management は、長期的な合計点最大化に寄与する判断を優先してください。

- Kaggle CLI/API upload/submit は、ユーザーの明示指示または dedicated official submit/score workflow からのみ使ってください。
- 通常の solve/build/score/cost experiment workflow では submission しないでください。
- dependency と virtual environment は `uv` で管理してください。
- dependency は `pyproject.toml` に記録し、`requirements.txt` は作らないでください。
- generated artifacts は原則 git 管理しないでください。
- 1 task ずつ build/score してください。
- `solutions/taskNNN.py` は `build_model() -> onnx.ModelProto` を提供する static ONNX graph generator です。
- 任意の Python runtime solver を ONNX に変換する設計は禁止です。

## Repository Management Policy

この repo では、疎結合性・整合性・整理整頓を重視してください。
目的は、設計の拡張性、長期的な score-up 効率、蓄積した知見の参照性を保つことです。

- workflow / skill / tool / data file の責務を混ぜないでください。
- 同じ意味の情報を複数箇所で手管理しないでください。
- tracked file は source、durable knowledge、curated history に限定してください。
- raw artifact、generated artifact、temporary dump は原則 ignored path に置いてください。
- task commit、note-only commit、workflow/tooling commit は分けてください。
- 共通方針は `AGENTS.md` を source of truth とし、各 skill に重複コピーしないでください。

## Source-of-Truth Boundaries

- `.agents/skills/*/SKILL.md`: workflow 固有の手順、trigger、責務境界、委譲先。
- `docs/`: task 横断で再利用する安定知識、public facts、official observations。
- `experiments/taskNNN.md`: task 固有の仮説、失敗知見、promotion 履歴。
- `task_ledger.json` / `task_ledger.md`: task 状態と local/official score の台帳。
- `local_score_progress.json` / `local_score_progress.md`: 改善作業中の local estimate 現在地履歴。
- `official_score_snapshots.json` / `official_score_snapshots.md`: official submit で確認した aggregate score snapshot 履歴。
- `outputs/` / `submissions/`: raw/generated artifacts。原則 git 管理しない。

## Adding New AI Workflows

新しい skill / tool / tracked data file を追加する前に、既存の責務境界に入るか、
新しい境界が必要かを明示してください。

- 新しい skill は、trigger、owned files、read-only source、delegation 先、
  commit 対象、ignored artifact 置き場を短く定義してください。
- 知見体系化 workflow は、task 横断の安定知識を `docs/` に、task 固有の実験知見を
  `experiments/taskNNN.md` に置いてください。
- 他 repo 情報抽出 workflow は、抽出元 repo / commit / URL / timestamp を記録し、
  未検証の外部情報を official facts と混ぜないでください。
- 外部由来の raw dump は tracked にせず、必要な要約・出典・判断だけを
  tracked knowledge にしてください。

`AGENTS.md` には、変動する score ranking、一時的な作業状態、task 固有の攻略仮説、
raw logs、各 skill の詳細手順を混ぜないでください。
