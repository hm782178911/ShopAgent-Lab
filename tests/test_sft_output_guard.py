import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('occupied', ['adapter', 'merged'])
def test_sft_refuses_nonempty_outputs_before_loading_model(tmp_path, occupied):
    target = tmp_path / occupied
    target.mkdir()
    sentinel = target / 'keep.txt'
    sentinel.write_text('preserve')
    env = os.environ | {'BASE_MODEL': 'not-a-real-model',
                        'SFT_ADAPTER_DIR': str(tmp_path / 'adapter'),
                        'SFT_MERGED_DIR': str(tmp_path / 'merged')}
    result = subprocess.run(['bash', str(ROOT / 'scripts/sft.sh'), '--max-steps', '1'],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'SFT output must be new or empty' in result.stderr
    assert sentinel.read_text() == 'preserve'
