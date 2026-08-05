#!/usr/bin/env bash
# Daily backup of the SQLite DB to a timestamped, gzipped snapshot.
# Add to crontab on the VPS, e.g. once a night:
#   5 3 * * *  /opt/jufo-transit/scripts/backup.sh >> /var/log/jufo-backup.log 2>&1
#
# For real off-box safety, sync BACKUP_DIR to object storage (rclone/S3) too.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB="$ROOT/data/transit.db"
BACKUP_DIR="$ROOT/data/backups"
mkdir -p "$BACKUP_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/transit_${STAMP}.db"

# Use sqlite3's online backup so we get a consistent copy even while WAL is active.
sqlite3 "$DB" ".backup '$OUT'"
gzip -f "$OUT"
echo "backup written: ${OUT}.gz"

# Keep the last 14 daily snapshots on-box; rely on off-box sync for the rest.
ls -1t "$BACKUP_DIR"/transit_*.db.gz | tail -n +15 | xargs -r rm -f
