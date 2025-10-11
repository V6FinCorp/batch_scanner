import os
import sys
import json

# Ensure repository package dir is on sys.path so imports work during pytest
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scanner_manager import ScannerManager


def test_run_scanner_errors_when_config_missing(tmp_path, monkeypatch):
    # Setup: point to a temp config dir
    sm = ScannerManager()
    cfg_dir = os.path.join(sm.scanners_dir, 'config')
    # Backup original files if present
    ema_cfg = os.path.join(cfg_dir, 'ema_config.json')
    backup = None
    if os.path.exists(ema_cfg):
        backup = ema_cfg + '.bak'
        os.rename(ema_cfg, backup)

    try:
        # Create an empty/invalid ema_config.json
        with open(ema_cfg, 'w', encoding='utf-8') as f:
            json.dump({}, f)

        res = sm.run_scanner('ema', ['ADANIENT'], '15mins', 2)
        assert isinstance(res, dict)
        assert res.get('category') == 'configuration' or 'error' in res

    finally:
        # Restore backup
        if backup and os.path.exists(backup):
            os.remove(ema_cfg)
            os.rename(backup, ema_cfg)


def test_manager_uses_config_periods(monkeypatch):
    sm = ScannerManager()
    cfg_file = os.path.join(sm.scanners_dir, 'config', 'ema_config.json')
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    assert 'ema_periods' in cfg and isinstance(cfg['ema_periods'], list)
    # The manager's run_scanner reads the config; just ensure validate_config no longer fails when kwargs omit emaPeriods
    res = sm.validate_config('ema', ['ADANIENT'], '15mins', 2)
    assert res.get('valid') is True
