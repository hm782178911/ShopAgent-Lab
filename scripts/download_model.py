#!/usr/bin/env python3
"""Download the model selected in configs/model.yaml, or resolve its local path."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopping_grpo.model_source import resolve_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--local-only', action='store_true')
    args=p.parse_args()
    print(resolve_model(args.config, download=not args.local_only))

if __name__=='__main__': main()
