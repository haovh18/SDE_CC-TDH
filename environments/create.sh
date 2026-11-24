#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-all}"

create_one() {
  local name="$1"
  conda env create -f "$root/environments/$name.yaml"
  conda run -n "sde-cctdh-${name/temporal_experts/temporal}" \
    python -m pip install -e "$root"
}

case "$profile" in
  qwen|imagebind|temporal_experts)
    create_one "$profile"
    ;;
  all)
    create_one qwen
    create_one imagebind
    create_one temporal_experts
    ;;
  *)
    printf 'usage: %s {qwen|imagebind|temporal_experts|all}\n' "$0" >&2
    exit 2
    ;;
esac

