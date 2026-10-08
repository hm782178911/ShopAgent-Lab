import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

from scripts import merge_lora_adapter

ROOT = Path(__file__).resolve().parents[1]


def load_new_module(path):
    assert path.is_file(), f"Missing implementation: {path.name}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_rejects_other_sglang_commit(monkeypatch):
    runtime = load_new_module(ROOT / 'src/shopping_grpo/runtime.py')
    expected = {'sglang': '0.5.15.post2.dev468+g85ac56c82', 'torch': '2.11.0'}
    installed = dict(expected, torch='2.11.0+cu130')
    monkeypatch.setattr(runtime, 'version', installed.__getitem__)
    assert runtime.validate_versions(expected) == installed
    installed['sglang'] = '0.5.15.post2.dev468+g00000000'
    with pytest.raises(RuntimeError, match='sglang'):
        runtime.validate_versions(expected)


@pytest.mark.parametrize('old,tokenizer,expected', [
    (248044, 248046, [248044, 248046]),
    ([248044, 248046], 248046, [248044, 248046]),
    (None, 248046, [248046]),
    (None, None, None),
])
def test_merge_preserves_eos_and_adds_chat_terminator(old, tokenizer, expected):
    assert hasattr(merge_lora_adapter, 'merge_eos_token_ids')
    assert merge_lora_adapter.merge_eos_token_ids(old, tokenizer) == expected


def test_patch_is_repeatable_and_does_not_modify_hardlinked_source(tmp_path):
    patcher = load_new_module(ROOT / 'scripts/apply_verl_sglang_patches.py')
    original = b'value = 1\n'
    patched = b'value = 2\n'
    source = tmp_path / 'cache.py'
    target = tmp_path / 'installed.py'
    source.write_bytes(original)
    os.link(source, target)
    patch_file = tmp_path / 'fix.patch'
    patch_file.write_text('--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n')
    hashes = [hashlib.sha256(b).hexdigest() for b in (original, patched)]
    patcher.apply_verified_patch(target, patch_file, *hashes)
    patcher.apply_verified_patch(target, patch_file, *hashes)
    assert target.read_bytes() == patched
    assert source.read_bytes() == original
    target.write_bytes(b'unknown = True\n')
    with pytest.raises(RuntimeError, match='unknown'):
        patcher.apply_verified_patch(target, patch_file, *hashes)
    assert target.read_bytes() == b'unknown = True\n'
