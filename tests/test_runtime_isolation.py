from pathlib import Path

import pytest

from shopping_grpo.runtime import validate_isolation, prepare_cuda_layout


def environment(tmp_path, inherited=False):
    prefix = tmp_path / '.venv'
    prefix.mkdir()
    (prefix / 'pyvenv.cfg').write_text(
        f'include-system-site-packages = {str(inherited).lower()}\n'
    )
    return prefix


def test_accepts_only_own_site_packages(tmp_path):
    prefix = environment(tmp_path)
    validate_isolation(prefix, [str(prefix / 'lib/python3.12/site-packages')])


def test_rejects_inherited_environment(tmp_path):
    prefix = environment(tmp_path, inherited=True)
    with pytest.raises(RuntimeError, match='system-site-packages'):
        validate_isolation(prefix, [])


def test_rejects_external_packages_injected_on_path(tmp_path):
    prefix = environment(tmp_path)
    with pytest.raises(RuntimeError, match='External'):
        validate_isolation(prefix, ['/opt/conda/lib/python3.12/site-packages'])


def test_rejects_linked_environment(tmp_path):
    prefix = environment(tmp_path)
    link = tmp_path / 'linked'
    link.symlink_to(prefix, target_is_directory=True)
    with pytest.raises(RuntimeError, match='linked'):
        validate_isolation(link, [])


def test_cuda_wheel_layout_supports_standard_linker_names(tmp_path):
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'libcudart.so.13').write_bytes(b'cuda runtime')
    prepare_cuda_layout(tmp_path)
    prepare_cuda_layout(tmp_path)
    assert (tmp_path / 'lib64/libcudart.so').read_bytes() == b'cuda runtime'


def test_cuda_layout_refuses_external_existing_link(tmp_path):
    (tmp_path / 'lib').mkdir()
    (tmp_path / 'lib/libcudart.so.13').touch()
    (tmp_path / 'lib64').symlink_to('/outside/cuda/lib64')
    with pytest.raises(RuntimeError, match='Unexpected'):
        prepare_cuda_layout(tmp_path)
