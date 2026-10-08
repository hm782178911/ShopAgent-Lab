"""The shared runtime contract for setup and GRPO, without GPU initialization."""

import argparse
import json
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def load_profile():
    path = Path(__file__).resolve().parents[2] / 'configs/runtime.json'
    return json.loads(path.read_text(encoding='utf-8'))


def validate_isolation(prefix=None, search_path=None):
    """Refuse inherited or injected third-party package directories."""
    prefix = Path(sys.prefix if prefix is None else prefix)
    search_path = sys.path if search_path is None else search_path
    if prefix.is_symlink():
        raise RuntimeError(f'Refusing linked environment: {prefix}')
    config = prefix / 'pyvenv.cfg'
    if not config.is_file():
        raise RuntimeError(f'An isolated virtual environment is required: {prefix}')
    values = dict(line.strip().split('=', 1) for line in config.read_text().splitlines() if '=' in line)
    values = {key.strip(): value.strip().lower() for key, value in values.items()}
    if values.get('include-system-site-packages') != 'false':
        raise RuntimeError('include-system-site-packages must be false')
    for entry in search_path:
        path = Path(entry).resolve()
        if {'site-packages', 'dist-packages'} & set(path.parts) and not path.is_relative_to(prefix.resolve()):
            raise RuntimeError(f'External package directory on Python path: {path}')


def prepare_cuda_layout(cuda_root):
    """Provide the standard CUDA linker layout for NVIDIA's cu13 wheels."""
    cuda_root = Path(cuda_root)
    runtime = cuda_root / 'lib/libcudart.so.13'
    if not runtime.is_file():
        raise RuntimeError(f'CUDA runtime wheel is missing: {runtime}')
    for link, target in [(cuda_root / 'lib64', 'lib'),
                         (cuda_root / 'lib/libcudart.so', 'libcudart.so.13')]:
        expected = (link.parent / target).resolve()
        if link.is_symlink() or link.exists():
            if link.resolve() != expected:
                raise RuntimeError(f'Unexpected existing CUDA linker path: {link}')
        else:
            link.symlink_to(target, target_is_directory=expected.is_dir())


def validate_versions(expected):
    installed = {}
    for name, required in expected.items():
        try:
            actual = version(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f'Missing runtime dependency: {name}=={required}') from exc
        # CUDA build suffixes are allowed; the SGLang source commit is pinned.
        comparable = actual if '+' in required else actual.split('+', 1)[0]
        if comparable != required:
            raise RuntimeError(f'Incompatible runtime: expected {name}=={required}, got {actual}')
        installed[name] = actual
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-only', action='store_true')
    parser.add_argument('--prepare-cuda', action='store_true')
    args = parser.parse_args()
    profile = load_profile()
    if f'{sys.version_info.major}.{sys.version_info.minor}' != profile['python']:
        raise SystemExit(f"Python {profile['python']} is required; got {sys.version.split()[0]}")
    header = Path(sysconfig.get_path('include')) / 'Python.h'
    if not header.is_file():
        raise SystemExit(f'Python development header missing: {header}')
    expected = profile['packages']
    if args.base_only:
        expected = {name: expected[name] for name in profile['base_packages']}
    try:
        validate_isolation()
        installed = validate_versions(expected)
        if args.prepare_cuda:
            prepare_cuda_layout(Path(sysconfig.get_path('purelib')) / 'nvidia/cu13')
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({'python': sys.executable, 'packages': installed}, indent=2))


if __name__ == '__main__':
    main()
