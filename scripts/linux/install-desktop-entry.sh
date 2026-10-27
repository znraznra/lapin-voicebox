#!/usr/bin/env bash
# Add "Voicebox" to the app menu (GNOME, COSMIC, KDE, ...). Remove with --uninstall.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ENTRY="$APPS/voicebox.desktop"

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$ENTRY"; echo "Removed $ENTRY"; exit 0
fi

ICON="$REPO/tauri/src-tauri/icons/128x128.png"
[ -f "$ICON" ] || ICON="voicebox"
chmod +x "$REPO/scripts/linux/voicebox-launcher.sh"
mkdir -p "$APPS"
cat >"$ENTRY" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Voicebox
Comment=Local AI voice studio
Exec="$REPO/scripts/linux/voicebox-launcher.sh"
Icon=$ICON
Terminal=false
Categories=AudioVideo;Audio;
StartupNotify=false
DESKTOP
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true
echo "Installed $ENTRY"
echo "Search for 'Voicebox' in your app launcher. Logs: ~/.local/state/voicebox/launcher.log"
