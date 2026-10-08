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
        installed = validate_versions(expected)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({'python': sys.executable, 'packages': installed}, indent=2))


if __name__ == '__main__':
    main()
