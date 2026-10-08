from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_missing_checkpoint_has_actionable_local_error(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/export_grpo.sh'),
                             str(tmp_path / 'missing'), str(tmp_path / 'out')],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert 'checkpoint is missing' in result.stderr
    assert not (tmp_path / 'out').exists()


def test_export_refuses_existing_output_without_modifying_it(tmp_path):
    checkpoint = tmp_path / 'actor'
    (checkpoint / 'huggingface').mkdir(parents=True)
    (checkpoint / 'huggingface/config.json').write_text('{}')
    output = tmp_path / 'out'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('keep me')
    result = subprocess.run(['bash', str(ROOT / 'scripts/export_grpo.sh'),
                             str(checkpoint), str(output)], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'output must be new or empty' in result.stderr
    assert sentinel.read_text() == 'keep me'
