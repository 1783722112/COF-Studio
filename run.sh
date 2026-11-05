#!/usr/bin/env bash
# COF Studio —— macOS / Linux 启动
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "  [错误] 没找到 Python，请先运行 ./install.sh"
  exit 1
fi

exec "$PY" cof_studio.py --open
