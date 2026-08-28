#!/bin/sh
# Snapshot the live Fly DB into riai/riai.db for local testing.
# VACUUM INTO (not a file copy) — the DB is WAL mode, so a raw copy would
# miss uncommitted-to-main data and can tear mid-checkpoint.
set -e
fly ssh console -C "python -c \"import sqlite3,os; os.path.exists('/tmp/s.db') and os.remove('/tmp/s.db'); sqlite3.connect('/data/riai.db').execute(\\\"VACUUM INTO '/tmp/s.db'\\\")\""
fly ssh sftp get /tmp/s.db riai/riai.db
echo "pulled $(du -h riai/riai.db | cut -f1) -> riai/riai.db"
