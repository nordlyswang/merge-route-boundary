#!/usr/bin/env python
"""Download model resources declared in configs/resources/models.yaml."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "resources" / "models.yaml"


def default_model_root() -> Path:
    return Path(os.environ.get("MRB_MODEL_ROOT", REPO_ROOT / "models")).expanduser()


def default_hf_cache() -> Path:
    return Path(
        os.environ.get(
            "HF_HUB_CACHE",
            os.environ.get("HF_HOME", str(REPO_ROOT / "cache" / "huggingface")) + "/hub",
        )
    ).expanduser()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"{path} must contain a 'models' list")
    return models


def selected_resources(resources: list[dict[str, Any]], include_all: bool) -> list[dict[str, Any]]:
    if include_all:
        return resources
    return [resource for resource in resources if resource.get("required") is True]


def local_model_dir(model_root: Path, repo_id: str) -> Path:
    return model_root / "huggingface" / repo_id.replace("/", "__")


def download_model(
    resource: dict[str, Any],
    model_root: Path,
    cache_dir: Path,
    include_weights: bool,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for model downloads") from exc

    repo_id = resource["id"]
    allow_patterns = None if include_weights else resource.get("allow_patterns", [])
    snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        local_dir=str(local_model_dir(model_root, repo_id)),
        allow_patterns=allow_patterns,
        token=os.environ.get("HF_TOKEN"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-root", type=Path, default=default_model_root())
    parser.add_argument("--hf-cache", type=Path, default=default_hf_cache())
    parser.add_argument("--all", action="store_true", help="Include optional resources.")
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Download full snapshots instead of metadata/tokenizer/config files only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resources = selected_resources(load_manifest(args.manifest), args.all)
    model_root = args.model_root.expanduser()
    hf_cache = args.hf_cache.expanduser()
    mode = "full snapshot" if args.include_weights else "metadata_only"

    print(f"Model manifest: {args.manifest}")
    print(f"Model root: {model_root}")
    print(f"HF hub cache: {hf_cache}")
    print(f"Mode: {mode}")
    if not os.environ.get("MRB_MODEL_ROOT"):
        print("WARN MRB_MODEL_ROOT is not set; using repository-local ignored models/ fallback.")
    if not os.environ.get("HF_HUB_CACHE"):
        print("WARN HF_HUB_CACHE is not set; using repository-local ignored cache/ fallback.")

    for resource in resources:
        repo_id = resource.get("id")
        source = resource.get("source")
        patterns = None if args.include_weights else resource.get("allow_patterns", [])
        print(f"{'[dry-run] ' if args.dry_run else ''}{repo_id} ({source}) patterns={patterns}")
        if args.dry_run:
            continue
        if source != "huggingface":
            raise ValueError(f"Unsupported model source for {repo_id}: {source}")
        model_root.mkdir(parents=True, exist_ok=True)
        hf_cache.mkdir(parents=True, exist_ok=True)
        download_model(resource, model_root, hf_cache, args.include_weights)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
