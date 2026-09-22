"""Resolve the configured ModelScope snapshot without implicit network access."""
from pathlib import Path
import hashlib
import json
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=None):
    path = Path(path or ROOT / "configs/model.yaml").expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("source") != "modelscope":
        raise ValueError("model source must be modelscope")
    for key in ("model_id", "revision", "local_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"model config requires nonempty {key}")
    target = Path(config["local_dir"]).expanduser()
    config["local_dir"] = target if target.is_absolute() else path.parent.parent / target
    return config

def validate_weights(target):
    if not (target / "config.json").is_file():
        raise FileNotFoundError(f"Missing {target}/config.json; run scripts/download_model.py")
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (target / name).is_file() and (target / name).stat().st_size > 0:
            return
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = target / name
        if index.is_file():
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
            if shards and all((target / shard).is_file() and (target / shard).stat().st_size > 0 for shard in shards):
                return
    raise FileNotFoundError(f"No complete model weights in {target}")

def resolve_model(path=None, download=False):
    config = load_config(path)
    target = config["local_dir"].resolve()
    identity = {k: config[k] for k in ("source", "model_id", "revision")}
    marker = target / "model-source.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text())
        if any(recorded.get(k) != v for k,v in identity.items()):
            raise ValueError("Model provenance differs from config; choose a new local_dir")
        validate_weights(target)
        return target
    if not download:
        validate_weights(target)
        raise ValueError("Missing model provenance; use download_model.py with an empty local_dir")
    pending = target / ".download-source.json"
    if pending.is_file():
        if json.loads(pending.read_text()) != identity:
            raise ValueError("Partial download provenance differs; choose a new local_dir")
    elif target.exists() and any(target.iterdir()):
        raise ValueError("Untracked nonempty model directory; choose a new local_dir")
    target.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(identity))
    from modelscope import snapshot_download
    # The library can print progress; keep stdout reserved for the returned path.
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        snapshot_download(model_id=config["model_id"], revision=config["revision"], local_dir=str(target))
    validate_weights(target)
    hashes = {}
    for file in sorted(target.rglob("*")):
        if file.is_file() and not any(part.startswith('.') for part in file.relative_to(target).parts):
            with file.open('rb') as stream:
                hashes[str(file.relative_to(target))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    marker.write_text(json.dumps({**identity, "files_sha256": hashes}, indent=2)+'\n')
    pending.unlink(missing_ok=True)
    return target
