#!/usr/bin/env bash
set -euo pipefail
cd /root/task
python -m pip install -q -r requirements.txt
python -m agent --selfcheck
printf '%s\n' 'ready'
