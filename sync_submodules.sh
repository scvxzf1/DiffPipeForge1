#!/usr/bin/env bash

set -euo pipefail

# 说明：可通过参数或环境变量控制最大重试次数。
# - 传参优先：sync_submodules.sh 10
# - 环境变量：SUBMODULE_MAX_RETRIES=10 sync_submodules.sh
# - 默认为 0（无限重试）
MAX_RETRIES="${1:-${SUBMODULE_MAX_RETRIES:-0}}"

if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "[sync_submodules] 错误：最大重试次数必须是非负整数，当前值: $MAX_RETRIES" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

attempt=1
while true; do
  echo "[sync_submodules] 第 ${attempt} 次尝试同步子模块..."

  if git submodule sync --recursive && git submodule update --init --recursive --jobs 8; then
    echo "[sync_submodules] 子模块同步完成。"
    exit 0
  fi

  if [[ "$MAX_RETRIES" -gt 0 && "$attempt" -ge "$MAX_RETRIES" ]]; then
    echo "[sync_submodules] 已达到最大重试次数 (${MAX_RETRIES})，同步失败。" >&2
    exit 1
  fi

  echo "[sync_submodules] 同步失败，3 秒后重试..." >&2
  sleep 3
  attempt=$((attempt + 1))
done
