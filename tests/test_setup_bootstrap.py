import os
import shutil
import subprocess
from pathlib import Path


def test_existing_uv_platform_suffix_does_not_trigger_download(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'scripts').mkdir()
    shutil.copyfile(root / 'scripts/setup.sh', tmp_path / 'scripts/setup.sh')
    data = tmp_path / 'environments/ShopSimulator/shop_env/data'
    data.mkdir(parents=True)
    (data / 'fine_items_eval_train_all.json.gz').touch()
    tools = tmp_path / 'bin'
    tools.mkdir()
    uv = tools / 'uv'
    uv.write_text('#!/bin/bash\nif [[ "$1" == "--version" ]]; then echo "uv 0.11.30 (x86_64-unknown-linux-gnu)"; else exit 73; fi\n')
    uv.chmod(0o755)
    curl = tools / 'curl'
    curl.write_text('#!/bin/bash\necho UNEXPECTED_DOWNLOAD\nexit 74\n')
    curl.chmod(0o755)
    result = subprocess.run(['bash', str(tmp_path / 'scripts/setup.sh')],
                            env=os.environ | {'PATH': str(tools) + ':/usr/bin:/bin', 'UV_OFFLINE': 'true'},
                            text=True, capture_output=True)
    assert result.returncode == 73, result.stdout + result.stderr
    assert 'UNEXPECTED_DOWNLOAD' not in result.stdout
