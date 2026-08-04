#!/usr/bin/env zsh

set -u

PORT="7862"
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
elif [[ -x "/Users/xieyulong/Documents/Codex/语音srt生成项目/.venv/bin/python" ]]; then
  PYTHON_BIN="/Users/xieyulong/Documents/Codex/语音srt生成项目/.venv/bin/python"
else
  echo "错误：找不到可用的 Python 解释器。"
  echo "请先在项目中创建 .venv 并安装 requirements.txt。"
  exit 1
fi

listener_pids() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null
}

old_pids=("${(@f)$(listener_pids)}")
if [[ -n "${old_pids[1]-}" ]]; then
  echo "发现 7862 端口已有服务，正在关闭：${old_pids[*]}"
  for pid in $old_pids; do
    [[ "$pid" == <-> ]] && kill -TERM "$pid" 2>/dev/null || true
  done

  for _ in {1..50}; do
    sleep 0.1
    old_pids=("${(@f)$(listener_pids)}")
    [[ -z "${old_pids[1]-}" ]] && break
  done

  old_pids=("${(@f)$(listener_pids)}")
  if [[ -n "${old_pids[1]-}" ]]; then
    echo "服务未正常退出，强制关闭：${old_pids[*]}"
    for pid in $old_pids; do
      [[ "$pid" == <-> ]] && kill -KILL "$pid" 2>/dev/null || true
    done

    old_pids=("${(@f)$(listener_pids)}")
    if [[ -n "${old_pids[1]-}" ]]; then
      echo "错误：无法关闭 7862 端口上的旧服务，请检查进程权限。"
      exit 1
    fi
  fi
fi

cd "$PROJECT_DIR" || exit 1
echo "启动服务器：http://127.0.0.1:$PORT"
exec env PORT="$PORT" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" app.py
