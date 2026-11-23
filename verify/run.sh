#!/usr/bin/env bash
# usage: run.sh <provider> <model-id> <label>
set -u
PROV="$1"; MODEL="$2"; LABEL="$3"
V=$HOME/projects/eta/verify
W=$V/work-$LABEL
rm -rf "$W" && mkdir -p "$W" && cp $V/TASK.md "$W/"
export PI_CODING_AGENT_DIR=$V/pi-agent PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0
export ETA_HOME=$V/eta-home-$LABEL
rm -rf "$ETA_HOME"
cd "$W"
P="pi --provider $PROV --model $MODEL"
echo "=== TURN 1 $(date +%T)"
$P -p "Start the task in TASK.md. Do items 1 and 2 (word_count and top_words) in textstats.py first, then stop there: I will check in with you, and after that you continue with the rest of the task." 2>&1 | tail -15
echo "=== TURN 2 /skill:eta $(date +%T)"
$P -c -p "/skill:eta" 2>&1 | tail -40
echo "=== TURN 3 $(date +%T)"
$P -c -p "Thanks. Reply with just: ok" 2>&1 | tail -5
echo "=== END $(date +%T)"
echo "=== files"; ls "$W"
echo "=== tests"; (cd "$W" && python3 -m unittest 2>&1 | tail -3)
echo "=== eta log"; cat "$ETA_HOME/predictions.jsonl" 2>/dev/null || echo "NO PREDICTIONS LOGGED"
echo "=== stats"; python3 $HOME/projects/eta/skills/eta/scripts/eta.py stats
