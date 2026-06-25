from __future__ import annotations

from pathlib import Path

from mrb.data.registry import check_project_data_symlink


def test_full_data_symlink_status(tmp_path: Path) -> None:
    data_root = tmp_path / "datasets"
    project_data = tmp_path / "project" / "data"
    data_root.mkdir()
    project_data.parent.mkdir()
    project_data.symlink_to(data_root, target_is_directory=True)

    status = check_project_data_symlink(project_data, data_root)

    assert status["mode"] == "full_symlink"
    assert status["is_symlink"] is True
    assert status["resolves_to_data_root"] is True
    assert status["warnings"] == []


def test_child_link_directory_is_reported(tmp_path: Path) -> None:
    data_root = tmp_path / "datasets"
    project_data = tmp_path / "project" / "data"
    (data_root / "torchvision").mkdir(parents=True)
    project_data.mkdir(parents=True)
    (project_data / "torchvision").symlink_to(data_root / "torchvision", target_is_directory=True)

    status = check_project_data_symlink(project_data, data_root)

    assert status["mode"] == "child_symlinks"
    assert status["is_symlink"] is False
    assert "torchvision" in status["child_symlinks"]
    assert status["warnings"]


def test_missing_project_data_is_nonfatal(tmp_path: Path) -> None:
    status = check_project_data_symlink(tmp_path / "project" / "data", tmp_path / "datasets")

    assert status["mode"] == "missing"
    assert status["exists"] is False
