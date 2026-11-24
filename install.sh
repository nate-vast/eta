#!/usr/bin/env bash
# Install the /eta skill for every agent harness on this machine.
#
#   ./install.sh                 symlink into every detected skills folder
#   ./install.sh --copy          copy instead of symlink (for machines without this repo)
#   ./install.sh --dir PATH      also install into PATH/eta (any other harness's skills folder)
#   ./install.sh --uninstall     remove what this script installed (keeps your ~/.eta history)
#
# Skill folders (Agent Skills format, SKILL.md):
#   ~/.claude/skills   Claude Code (CLI, desktop, T3 Code, Agent SDK)
#   ~/.agents/skills   Codex, pi, OpenCode, Gemini CLI and other tools that read the shared folder
#   ~/.cursor/skills, ~/.gemini/skills, ~/.config/opencode/skills  only if that tool is present
#                      and does not already read ~/.agents/skills
# The script itself is linked at ~/.eta/eta.py so every harness uses one stable path.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$REPO/skills/eta"
MODE=link
EXTRA=()
UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --copy) MODE=copy ;;
    --dir) EXTRA+=("$2"); shift ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

targets=("$HOME/.claude/skills" "$HOME/.agents/skills")
[ -d "$HOME/.cursor" ] && targets+=("$HOME/.cursor/skills")
for d in "${EXTRA[@]+"${EXTRA[@]}"}"; do targets+=("$d"); done

ours() {  # is $1 something this installer put there?
  [ -L "$1" ] && [ "$(readlink "$1")" = "$SRC" ] && return 0
  [ -f "$1/SKILL.md" ] && grep -q '^name: eta$' "$1/SKILL.md" && [ -f "$1/.eta-installed" ] && return 0
  return 1
}

if [ "$UNINSTALL" = 1 ]; then
  for t in "${targets[@]}"; do
    if ours "$t/eta"; then rm -rf "$t/eta"; echo "removed $t/eta"; fi
  done
  if [ -L "$HOME/.eta/eta.py" ]; then rm "$HOME/.eta/eta.py"; echo "removed ~/.eta/eta.py (history kept in ~/.eta)"; fi
  exit 0
fi

python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || { echo "needs python3 >= 3.9" >&2; exit 1; }
chmod +x "$SRC/scripts/eta.py"

mkdir -p "$HOME/.eta"
if [ "$MODE" = link ]; then
  ln -sfn "$SRC/scripts/eta.py" "$HOME/.eta/eta.py"
else
  rm -f "$HOME/.eta/eta.py"; cp "$SRC/scripts/eta.py" "$SRC/scripts/job_history.py" "$HOME/.eta/"
fi
echo "script  ~/.eta/eta.py"

for t in "${targets[@]}"; do
  dest="$t/eta"
  if [ -e "$dest" ] || [ -L "$dest" ]; then
    if ! ours "$dest"; then echo "skip    $dest (something else is already there)"; continue; fi
    rm -rf "$dest"
  fi
  mkdir -p "$t"
  if [ "$MODE" = link ]; then
    ln -s "$SRC" "$dest"
  else
    cp -R "$SRC" "$dest"; touch "$dest/.eta-installed"
  fi
  echo "skill   $dest"
done

echo
echo "Done. Start a new agent session, then type /eta (Codex: \$eta or /skills, pi: /skill:eta)."
