#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 说明：
# - 不传参：默认无限重试子模块同步
# - 传 1 个参数：作为最大重试次数
if [[ "$#" -gt 1 ]]; then
  echo "用法：./start_with_submodules.sh [最大重试次数]" >&2
  exit 2
fi

if [[ "$#" -eq 1 ]]; then
  retry_limit="$1"
else
  retry_limit="${SUBMODULE_MAX_RETRIES:-0}"
fi

if ! [[ "$retry_limit" =~ ^[0-9]+$ ]]; then
  echo "错误：最大重试次数必须是非负整数，当前值: $retry_limit" >&2
  exit 2
fi

echo "[start_with_submodules] 先同步子模块，再启动主程序..."
"$SCRIPT_DIR/sync_submodules.sh" "$retry_limit"

echo "[start_with_submodules] 子模块已就绪，开始执行 start.sh"
exec "$SCRIPT_DIR/start.sh"
