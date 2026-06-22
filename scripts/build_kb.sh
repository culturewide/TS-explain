#!/usr/bin/env bash
set -euo pipefail
python -m knowledge_base.build_kb --reset "$@"

