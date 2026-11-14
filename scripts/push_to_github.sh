#!/usr/bin/env bash
# One-shot first push of this repo to GitHub (macOS / Linux / Git Bash on Windows).
# Usage (from the project root):
#   bash scripts/push_to_github.sh                      # uses defaults below
#   bash scripts/push_to_github.sh samra761 flame-in-freefall
#
# Create the EMPTY repo on GitHub first (no README, no .gitignore, no license),
# or have the GitHub CLI installed and logged in (gh auth login) and it will be created for you.
set -euo pipefail
cd "$(dirname "$0")/.."

GH_USER="${1:-samra761}"
REPO="${2:-flame-in-freefall}"
REMOTE="https://github.com/${GH_USER}/${REPO}.git"

# 1. Identity (only set if missing)
git config --global user.name  >/dev/null || git config --global user.name  "Samra"
git config --global user.email >/dev/null || { echo "Set your email: git config --global user.email you@example.com"; exit 1; }

# 2. Init on main
[ -d .git ] || git init -b main

# 3. Sanity check: nothing over 50 MB is about to be committed
big=$(git ls-files --others --cached --exclude-standard -z | xargs -0 -I{} find "{}" -size +50M 2>/dev/null || true)
[ -z "$big" ] || { echo "Files over 50 MB would be committed:"; echo "$big"; exit 1; }

# 4. Stage + commit
git add .
git status --short
git commit -m "feat: initial commit of Flame in Freefall (The Blueprint 67)" \
           -m "AI-powered microgravity combustion analytics dashboard for NASA Space Apps.
SAME/FLEX/BASS-II/Saffire ingestion, Isolation Forest early warning,
LightGBM/RF material classifier, FLEX flame forecaster, Streamlit dashboard."

# 5. Remote (create via gh if available, else use the existing empty repo)
if command -v gh >/dev/null && ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "${GH_USER}/${REPO}" --public --source=. --remote=origin \
     --description "AI-powered microgravity combustion analytics, NASA Space Apps, The Blueprint 67" || true
fi
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$REMOTE"

# 6. Push
git branch -M main
git push -u origin main
echo "✔ Pushed: https://github.com/${GH_USER}/${REPO}"
