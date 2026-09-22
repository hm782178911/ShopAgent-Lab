#!/usr/bin/env python3
"""Run unchanged Final-200 Clean evaluation against configured SGLang serving."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shopping_grpo.model_source import load_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label',nargs='?',default='model')
    p.add_argument('--config',type=Path,default=os.getenv('MODEL_CONFIG'))
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not args.label or Path(args.label).name!=args.label or args.label in {'.','..'}:
        p.error('label must be a directory name')
    settings=load_config(args.config).get('serving',{})
    host=settings.get('host','127.0.0.1')
    if host=='0.0.0.0': host='127.0.0.1'
    url=os.getenv('LLM_BASE_URL',f'http://{host}:{settings.get("port",8000)}/v1')
    out=Path(os.getenv('EVAL_OUTPUT_DIR',str(ROOT/'outputs/evaluation'/args.label))).resolve()
    command=[sys.executable,str(ROOT/'scripts/evaluate_shop_benchmark.py'),
        '--benchmark',str(ROOT/'data/evaluation/tasks.jsonl'),
        '--output',str(out/'trajectories.jsonl'),'--summary',str(out/'summary.json'),
        '--base-url',os.getenv('SHOPSIM_BASE_URL','http://127.0.0.1:5700'),
        '--model',os.getenv('SERVED_MODEL_NAME',settings.get('name','shopping-agent')),
        '--llm-base-url',url,'--api-key',os.getenv('LLM_API_KEY','EMPTY'),
        '--context-window',str(settings.get('context_length',24576))]
    if args.dry_run:
        import json
        safe=command.copy(); safe[safe.index('--api-key')+1]='<redacted>'
        print(json.dumps(safe,indent=2)); return
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run(command,cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_eval_report.py'),'--run-dir',str(out)],cwd=ROOT,check=True)

if __name__=='__main__': main()
