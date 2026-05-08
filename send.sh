ssh runpod-server "[ -d /workspace/models ] || mkdir /workspace/models"
ssh runpod-server "[ -d /workspace/data ] || mkdir /workspace/data"
scp pyproject.toml runpod-server:/workspace
scp multi-head.py runpod-server:/workspace
scp data/* runpod-server:/workspace/data
