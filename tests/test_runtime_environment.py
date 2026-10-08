import os
import subprocess
import sys
from pathlib import Path


def test_launcher_selects_project_cuda_and_python(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/runtime_env.sh'
    assert script.is_file()
    venv = tmp_path / '.venv'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'bin/python').symlink_to(sys.executable)
    cuda = venv / 'lib/python3.12/site-packages/nvidia/cu13'
    (cuda / 'bin').mkdir(parents=True)
    (cuda / 'bin/nvcc').touch()
    env = dict(os.environ, ROOT=str(tmp_path), CUDA_HOME='/old/cuda',
               PYTHONPATH='/old/packages', PYTHONHOME='/old/python',
               LD_LIBRARY_PATH='/old/libraries')
    result = subprocess.run(
        ['bash', '-c', 'source "$1"; printf "%s\\n" "$CUDA_HOME" "$PATH" "${PYTHONPATH-unset}" "${PYTHONHOME-unset}" "${LD_LIBRARY_PATH-unset}"', 'test', str(script)],
        env=env, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert result[0] == str(cuda)
    assert result[1].startswith(f'{venv}/bin:{cuda}/bin:')
    assert '/old/' not in result[1]
    assert result[2:] == ['unset', 'unset', str(cuda / 'lib64')]
