#!/usr/bin/env python3
"""Apply the two pinned veRL/SGLang fixes without changing shared cache files."""

import argparse
import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.apply_verl_dynamic_sampling_patch import resolve_installed_ray_trainer

PATCHES = (
    ('workers/rollout/replica.py', 'verl-0.8.0-sglang-gpu-import.patch',
     '07796ad3a93a870c763f97a6ca65e2c4e9c8a059de1daf96674d1e889a19a7a7',
     '1d370668cb9c41308a3eb37049507dbf3fa9ee1f9cea7490c2c3c7b9550ea822'),
    ('workers/rollout/sglang_rollout/async_sglang_server.py',
     'verl-0.8.0-sglang-merged-rollout.patch',
     'b230f0af10f889cba1b6582a089f34b62bac5a7eb9cbf7dcf7987a0bd5ee8bd3',
     'fe0e106fce338bc8702e1b7d6484aa8ce832355db168cfe7c571081cdc016d0e'),
)


def apply_verified_patch(target, patch_file, original_sha, patched_sha, check_only=False):
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual == patched_sha:
        return
    if actual != original_sha:
        raise RuntimeError(f'Refusing unknown dependency source: {target}')
    if check_only:
        raise RuntimeError(f'Missing SGLang patch: {patch_file.name}; run scripts/setup.sh')
    # patch only a private temporary copy, then atomically replace the installed file.
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        private = Path(directory) / target.name
        shutil.copy2(target, private)
        subprocess.run(['patch', '--batch', '--forward', '--silent', str(private), str(patch_file)], check=True)
        if hashlib.sha256(private.read_bytes()).hexdigest() != patched_sha:
            raise RuntimeError(f'Patched hash mismatch: {target}')
        ast.parse(private.read_text(encoding='utf-8'))
        os.replace(private, target)


def verify_installation(check_only=False):
    # This guard checks both veRL version and ownership by this project's .venv.
    verl_root = resolve_installed_ray_trainer().parents[2]
    for relative, patch_name, original, patched in PATCHES:
        apply_verified_patch(verl_root / relative, ROOT / 'patches' / patch_name,
                             original, patched, check_only=check_only)
    print('Verified veRL SGLang patches')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        verify_installation(check_only=args.check)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == '__main__':
    main()
