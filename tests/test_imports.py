from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = [
    "check_env.py",
    "report_env.py",
    "download_datasets.py",
    "download_models.py",
    "verify_datasets.py",
    "report_storage.py",
    "report_data_storage.py",
    "verify_resources.py",
    "inspect_datasets.py",
    "build_split_manifest.py",
    "verify_task_streams.py",
    "create_feature_symlink.py",
    "estimate_feature_storage.py",
    "extract_features.py",
    "verify_feature_bank.py",
    "inspect_feature_bank.py",
    "compute_boundary_diagnostics.py",
    "inspect_boundary_diagnostics.py",
    "export_boundary_matrix.py",
    "smoke_check.py",
]


def import_script(path: Path) -> object:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_imports() -> None:
    import mrb

    assert mrb.__version__


def test_scripts_import_without_side_effects() -> None:
    for script_name in SCRIPT_NAMES:
        module = import_script(REPO_ROOT / "scripts" / script_name)
        assert hasattr(module, "main")
