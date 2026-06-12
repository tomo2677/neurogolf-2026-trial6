常に日本語で応答してください。ただし、code, command, file path, technical term は English のまま残してください。

この repo は NeuroGolf 2026 の local research / local validation 用です。

- Kaggle CLI/API upload/submit は、ユーザーの明示指示または dedicated official submit/score workflow からのみ使ってください。
- 通常の solve/build/score/cost experiment workflow では submission しないでください。
- dependency と virtual environment は `uv` で管理してください。
- dependency は `pyproject.toml` に記録し、`requirements.txt` は作らないでください。
- generated artifacts は原則 git 管理しないでください。
- 1 task ずつ build/score してください。
- `solutions/taskNNN.py` は `build_model() -> onnx.ModelProto` を提供する static ONNX graph generator です。
- 任意の Python runtime solver を ONNX に変換する設計は禁止です。
