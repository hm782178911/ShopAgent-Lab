import json
from pathlib import Path
import pytest
from shopping_grpo.model_source import load_config, resolve_model

def config(tmp_path):
    p=tmp_path/'configs'/'model.yaml'; p.parent.mkdir()
    p.write_text('source: modelscope\nmodel_id: Qwen/Qwen3.5-2B\nrevision: master\nlocal_dir: models/base\n')
    return p

def test_paths_relative_to_repo_not_cwd(tmp_path, monkeypatch):
    p=config(tmp_path); monkeypatch.chdir('/')
    assert load_config(p)['local_dir']==tmp_path/'models/base'

def test_incomplete_model_rejected(tmp_path):
    p=config(tmp_path)
    with pytest.raises(FileNotFoundError): resolve_model(p)

def test_cached_model_requires_matching_source(tmp_path):
    p=config(tmp_path); model=tmp_path/'models/base'; model.mkdir(parents=True)
    (model/'config.json').write_text('{}'); (model/'model.safetensors').write_bytes(b'weights')
    with pytest.raises(ValueError, match='provenance'): resolve_model(p)
    (model/'model-source.json').write_text(json.dumps({'source':'modelscope','model_id':'Qwen/Qwen3.5-2B','revision':'master'}))
    assert resolve_model(p)==model

def test_missing_weight_shard_rejected(tmp_path):
    p=config(tmp_path); model=tmp_path/'models/base'; model.mkdir(parents=True)
    (model/'config.json').write_text('{}')
    (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'missing.safetensors'}}))
    with pytest.raises(FileNotFoundError): resolve_model(p)

def test_download_records_source_and_can_resume(tmp_path, monkeypatch):
    import sys, types
    p=config(tmp_path); calls=[]
    def download(**kwargs):
        target=Path(kwargs['local_dir']); target.mkdir(parents=True,exist_ok=True)
        (target/'config.json').write_text('{}')
        calls.append(kwargs)
        if len(calls)==1: raise ConnectionError('temporary failure')
        (target/'model.safetensors').write_bytes(b'weights')
    monkeypatch.setitem(sys.modules,'modelscope',types.SimpleNamespace(snapshot_download=download))
    with pytest.raises(ConnectionError): resolve_model(p,download=True)
    path=resolve_model(p,download=True)
    record=json.loads((path/'model-source.json').read_text())
    assert record['model_id']=='Qwen/Qwen3.5-2B'
    assert len(record['files_sha256']['model.safetensors'])==64
    assert len(calls)==2
    assert resolve_model(p,download=True)==path
    assert len(calls)==2
