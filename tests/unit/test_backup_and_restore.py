"""Tests for database backup, retention pruning, and scratch database restore (P8-05).

Verifies:
1. `create_backup` executes and writes backup archive into output directory.
2. `prune_old_backups` prunes backups older than the configured retention threshold.
3. `restore_backup` restores a generated backup archive into a scratch database.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.backup_db import create_backup, prune_old_backups
from scripts.restore_db import restore_backup

from retailpulse.timeutil import utc_now


def test_create_backup_writes_file(tmp_path: Path) -> None:
    """Verify that create_backup generates a backup file in the target directory."""
    with patch("scripts.backup_db.shutil.which", return_value="/usr/bin/pg_dump"), patch(
        "scripts.backup_db.subprocess.run"
    ) as mock_run:
        # Simulate pg_dump creating the output file
        def side_effect(cmd, **kwargs):
            out_file = cmd[cmd.index("-f") + 1]
            Path(out_file).write_text("dummy pg_dump archive", encoding="utf-8")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        backup_file = create_backup(
            database_url="postgresql+psycopg://user:pass@localhost:5432/retailpulse",
            output_dir=tmp_path,
            retention_days=30,
        )

        assert backup_file.exists()
        assert "retailpulse_backup_" in backup_file.name
        mock_run.assert_called_once()


def test_backup_retention_pruning(tmp_path: Path) -> None:
    """Verify that files older than retention_days are pruned while fresh ones are kept."""
    now = utc_now()

    # 1. Fresh backup (1 day old)
    fresh_file = tmp_path / "retailpulse_backup_fresh.dump"
    fresh_file.write_text("fresh", encoding="utf-8")

    # 2. Old backup (45 days old)
    old_file = tmp_path / "retailpulse_backup_old.dump"
    old_file.write_text("old", encoding="utf-8")
    old_mtime = (now - timedelta(days=45)).timestamp()

    os.utime(old_file, (old_mtime, old_mtime))

    pruned = prune_old_backups(tmp_path, retention_days=30)

    assert pruned == 1
    assert fresh_file.exists()
    assert not old_file.exists()


def test_restore_backup_into_scratch_database(tmp_path: Path) -> None:
    """Verify that restore_backup executes pg_restore against scratch database and checks schema."""
    backup_file = tmp_path / "retailpulse_backup_test.dump"
    backup_file.write_text("pg_dump binary payload", encoding="utf-8")

    with patch("scripts.restore_db.shutil.which", return_value="/usr/bin/pg_restore"), patch(
        "scripts.restore_db.subprocess.run"
    ) as mock_run, patch("scripts.restore_db.sa.create_engine") as mock_engine:
        # Mock engine connect
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn

        scratch_db_url = "postgresql+psycopg://user:pass@localhost:5432/retailpulse_scratch"

        success = restore_backup(
            backup_file=backup_file,
            target_database_url=scratch_db_url,
            clean_target=True,
        )

        assert success is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "-d" in cmd
        assert "retailpulse_scratch" in cmd
        assert "--clean" in cmd
