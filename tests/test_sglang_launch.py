import importlib.util
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def test_serving_uses_local_checkpoint_and_custom_settings(tmp_path):
    spec=importlib.util.spec_from_file_location('serve', ROOT/'scripts/serve_sglang.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config=tmp_path/'model.yaml'
    config.write_text('source: modelscope\nmodel_id: Qwen/Qwen3.5-2B\nrevision: master\nlocal_dir: unused\nserving:\n  port: 8123\n  name: custom-agent\n  tensor_parallel_size: 2\n')
    model=tmp_path/'checkpoint with spaces'; model.mkdir()
    (model/'config.json').write_text('{}'); (model/'model.safetensors').write_bytes(b'weight')
    command=module.build_command(config,model)
    assert command[1:3]==['-m','sglang.launch_server']
    assert command[command.index('--model-path')+1]==str(model)
    assert command[command.index('--port')+1]=='8123'
    assert command[command.index('--tp-size')+1]=='2'

def test_eval_config_follows_serving_and_redacts_secret(tmp_path):
    import os
    config=tmp_path/'model.yaml'
    config.write_text('source: modelscope\nmodel_id: Qwen/Qwen3.5-2B\nrevision: master\nlocal_dir: unused\nserving:\n  port: 8123\n  name: custom-agent\n')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_configured.py'),'baseline','--config',str(config),'--dry-run'],capture_output=True,text=True,check=True,env={**os.environ,'LLM_API_KEY':'secret-test-value'})
    assert 'secret-test-value' not in result.stdout
    command=json.loads(result.stdout)
    assert command[command.index('--model')+1]=='custom-agent'
    assert command[command.index('--llm-base-url')+1]=='http://127.0.0.1:8123/v1'
