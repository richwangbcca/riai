"""Tests for config hot-reload (no network, no DB)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poller


def _write(p, text):
    p.write_text(text)
    # mtime has coarse resolution on some filesystems; force a distinct value
    st = p.stat()
    import os
    os.utime(p, (st.st_atime, st.st_mtime + 1))
    return p.stat().st_mtime


def test_reload_picks_up_edits(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("summarize:\n  model: old-model\n")
    cfg = poller._load_config(str(cfg_file))
    mtime = cfg_file.stat().st_mtime
    assert cfg["summarize"]["model"] == "old-model"

    _write(cfg_file, "summarize:\n  model: new-model\n")
    cfg, mtime = poller._maybe_reload_config(str(cfg_file), cfg, mtime)
    assert cfg["summarize"]["model"] == "new-model"


def test_unchanged_file_is_not_reparsed(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("a: 1\n")
    cfg = poller._load_config(str(cfg_file))
    mtime = cfg_file.stat().st_mtime

    sentinel = {"marker": "untouched"}
    got, got_mtime = poller._maybe_reload_config(str(cfg_file), sentinel, mtime)
    assert got is sentinel and got_mtime == mtime


def test_malformed_config_keeps_previous(tmp_path):
    """A broken edit must not take down a running poller."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("weights:\n  news: 0.5\n")
    good = poller._load_config(str(cfg_file))
    mtime = cfg_file.stat().st_mtime

    mtime_bad = _write(cfg_file, "weights:\n  news: [unclosed\n")
    cfg, mtime = poller._maybe_reload_config(str(cfg_file), good, mtime)
    assert cfg is good, "should have kept the last good config"
    # mtime advanced, so the broken file is not re-reported every iteration
    assert mtime == mtime_bad
    assert poller._maybe_reload_config(str(cfg_file), cfg, mtime)[0] is good

    # and it recovers once the file is valid again
    _write(cfg_file, "weights:\n  news: 0.7\n")
    cfg, mtime = poller._maybe_reload_config(str(cfg_file), cfg, mtime)
    assert cfg["weights"]["news"] == 0.7


def test_non_mapping_config_is_ignored(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("a: 1\n")
    good = poller._load_config(str(cfg_file))
    mtime = cfg_file.stat().st_mtime

    _write(cfg_file, "just a bare string\n")
    cfg, _ = poller._maybe_reload_config(str(cfg_file), good, mtime)
    assert cfg is good


def test_missing_file_keeps_previous(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("a: 1\n")
    good = poller._load_config(str(cfg_file))
    mtime = cfg_file.stat().st_mtime
    cfg_file.unlink()

    cfg, got_mtime = poller._maybe_reload_config(str(cfg_file), good, mtime)
    assert cfg is good and got_mtime == mtime
