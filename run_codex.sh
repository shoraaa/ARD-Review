#!/usr/bin/env bash
# Usage: ./run_codex.sh paper.pdf [--no_cal] [--model MODEL] [...]
set -euo pipefail
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 PAPER [review options...]" >&2
    exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paper="$1"
shift
exec "${PYTHON:-python}" "$repo_dir/code/main.py" --backend codex --single_paper "$paper" "$@"
