#!/bin/bash
# Collect Scholar data locally, then push only the validated snapshot branch.

set -euo pipefail

refresh_mode="incremental"
if [ "${1:-}" = "--full" ]; then
  refresh_mode="full"
elif [ -n "${1:-}" ]; then
  printf 'Usage: %s [--full]\n' "$0" >&2
  exit 64
fi

collector_root="${SCHOLAR_COLLECTOR_ROOT:-/Users/work/Codex/var}"
temporary_root="$collector_root/tmp/scholar-publications"
state_root="$collector_root/state"
lock_dir="$state_root/scholar-publication-collector.lock"
repository_url="https://github.com/cmccomb/scrape-my-publications.git"
snapshot_branch="automation/publication-snapshot"

mkdir -p "$temporary_root" "$state_root"
if ! mkdir "$lock_dir" 2>/dev/null; then
  printf 'Another Scholar publication collection is already running.\n' >&2
  exit 75
fi

run_directory="$(mktemp -d "$temporary_root/run.XXXXXX")"
cleanup() {
  rm -rf "$run_directory"
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

git clone --quiet "$repository_url" "$run_directory/repository"
cd "$run_directory/repository"

if git show-ref --verify --quiet "refs/remotes/origin/$snapshot_branch"; then
  git switch --create "$snapshot_branch" --track "origin/$snapshot_branch"
  git merge --no-edit origin/main
else
  git switch --create "$snapshot_branch" origin/main
fi

collector_args=(
  --dry-run
  --snapshot-path snapshots/publications.parquet
  --status-file status/last-collector.json
)
if [ "$refresh_mode" = "full" ]; then
  collector_args+=(--full-refresh)
fi

/Users/work/.local/bin/uv run \
  --with-requirements requirements.txt \
  python main.py "${collector_args[@]}"

git config user.name "The McComb Agency automation"
git config user.email "the.mccomb.agency@gmail.com"
git add \
  snapshots/publications.parquet \
  snapshots/publications.parquet.manifest.json
git add --force status/last-collector.json

if git diff --cached --quiet; then
  printf 'The validated publication snapshot is unchanged.\n'
  exit 0
fi

git commit --quiet -m "Collect $refresh_mode Scholar publication snapshot"
git push origin "HEAD:$snapshot_branch"
