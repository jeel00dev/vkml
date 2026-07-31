#!/bin/sh
# Build the documentation site and publish it to the gh-pages branch.
#
# Manual rather than CI-driven, deliberately: the build IMPORTS vkml to read
# signatures from the installed module, so it needs a built extension. A CI job
# that published without one would silently ship a site describing whatever
# stub it could import.
#
# The site lands on gh-pages rather than in /docs on main, so generated HTML
# never mixes with the design documents that live there.
set -eu

cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is dirty; commit or stash first" >&2
    exit 1
fi

SRC_REV=$(git rev-parse --short HEAD)
python web/build.py

WORK=$(mktemp -d)
cp -r web/_site/. "$WORK/"

git fetch origin gh-pages 2>/dev/null || true
if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
    git worktree add -f /tmp/vkml-pages gh-pages
else
    git worktree add -f --detach /tmp/vkml-pages
    git -C /tmp/vkml-pages checkout --orphan gh-pages
    git -C /tmp/vkml-pages rm -rf . >/dev/null 2>&1 || true
fi

find /tmp/vkml-pages -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -r "$WORK"/. /tmp/vkml-pages/
git -C /tmp/vkml-pages add -A
git -C /tmp/vkml-pages commit -q -m "docs: rebuild site from $SRC_REV" || {
    echo "no change since the last publish"; }
git -C /tmp/vkml-pages push origin gh-pages

git worktree remove --force /tmp/vkml-pages
rm -rf "$WORK"
echo "published; enable Pages on the gh-pages branch if it is not already"
