#!/usr/bin/env bash
# install.sh - symlink the npufast tools into a bin dir on PATH.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$HOME/.local/bin}"
mkdir -p "$BIN"
for t in npufast npufast-host npufast-bigmem; do
  if [[ -f "$DIR/bin/$t" ]]; then src="$DIR/bin/$t"; else src="$DIR/$t"; fi
  [[ -f "$src" ]] || { echo "skip: $t not found"; continue; }
  chmod +x "$src"; ln -sf "$src" "$BIN/$t"; echo "linked $BIN/$t -> $src"
done
case ":$PATH:" in
  *":$BIN:"*) echo "PATH ok";;
  *) echo "add to your shell rc:  export PATH=\"$BIN:\$PATH\"";;
esac
