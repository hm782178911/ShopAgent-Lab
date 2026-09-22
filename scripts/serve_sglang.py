#!/usr/bin/env python3
"""Serve the configured base model or an explicitly supplied merged checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopping_grpo.model_source import load_config, resolve_model, validate_weights

def build_command(config_path=None, model=None):
    config=load_config(config_path)
    path=Path(model).expanduser().resolve() if model else resolve_model(config_path)
    validate_weights(path)
    settings=config.get('serving', {})
    command=[sys.executable, '-m', 'sglang.launch_server', '--model-path', str(path)]
    for key, flag, default in [('name','served-model-name','shopping-agent'),
        ('host','host','127.0.0.1'), ('port','port',8000),
        ('context_length','context-length',24576),
        ('tensor_parallel_size','tp-size',1), ('mem_fraction_static','mem-fraction-static',0.7),
        ('tool_call_parser','tool-call-parser','qwen3_coder')]:
        command += ['--'+flag, str(settings.get(key,default))]
    command += ['--trust-remote-code']
    extra=settings.get('extra_args',[])
    if not isinstance(extra,list) or not all(isinstance(x,str) for x in extra):
        raise ValueError('serving.extra_args must be a list of strings')
    return command+extra

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',nargs='?',help='local merged SFT/GRPO model; defaults to model.yaml')
    p.add_argument('--config',type=Path,default=os.getenv('MODEL_CONFIG'))
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); command=build_command(a.config,a.model)
    if a.dry_run: print(json.dumps(command,indent=2)); return
    os.execv(command[0],command)

if __name__=='__main__': main()
