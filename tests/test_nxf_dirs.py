"""
The plugin must open even when the Nextflow base directory is unreachable.

A network drive that has gone away is not simply absent: on macOS the mount
point is left behind and every access raises OSError(EIO). That used to escape
widget construction, so the plugin could not be opened at all - including for
local runs that never needed the drive.
"""

import errno
from pathlib import Path

import pytest

from aiod_napari.inference.nxf import NxfWidget, path_available


@pytest.fixture
def dead_mount(tmp_path, monkeypatch):
    """
    A path that raises OSError(EIO) on access, like a disconnected drive.
    """
    dead = tmp_path / "mount"
    real_exists, real_mkdir = Path.exists, Path.mkdir

    def fail_if_under_mount(self, *args, real, **kwargs):
        if dead == self or dead in self.parents:
            raise OSError(errno.EIO, "Input/output error", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(
        Path, "exists", lambda self: fail_if_under_mount(self, real=real_exists)
    )
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda self, **kw: fail_if_under_mount(self, real=real_mkdir, **kw),
    )
    return dead


class DirStub:
    """
    The directory-handling half of NxfWidget, without needing a viewer.
    """

    setup_nxf_dir_cmd = NxfWidget.setup_nxf_dir_cmd
    _set_nxf_dirs = NxfWidget._set_nxf_dirs

    def __init__(self, default: Path):
        self.DEFAULT_BASE_DIR = default


def test_path_available_is_false_for_a_dead_mount(dead_mount):
    with pytest.raises(OSError, match="Input/output"):
        dead_mount.exists()  # what Path.exists() does, and why the guard exists
    assert path_available(dead_mount) is False


def test_path_available_reports_real_paths(tmp_path):
    assert path_available(tmp_path) is True
    assert path_available(tmp_path / "nope") is False


def test_unreachable_base_dir_falls_back_instead_of_raising(
    dead_mount, tmp_path, monkeypatch
):
    told = []
    monkeypatch.setattr("aiod_napari.inference.nxf.show_info", told.append)
    widget = DirStub(default=tmp_path / "fallback")

    widget.setup_nxf_dir_cmd(dead_mount / ".nextflow" / "run1")

    assert widget.nxf_base_dir == tmp_path / "fallback"
    assert widget.nxf_store_dir.exists()
    assert widget.nxf_work_dir.exists()
    # and the user is told, rather than it happening silently
    assert told and "falling back" in told[0]


def test_a_reachable_base_dir_is_used_as_given(tmp_path):
    widget = DirStub(default=tmp_path / "fallback")
    wanted = tmp_path / "wanted"

    widget.setup_nxf_dir_cmd(wanted)

    assert widget.nxf_base_dir == wanted
    assert widget.nxf_store_dir == wanted / "aiod_cache"
    assert widget.nxf_work_dir.exists()
    assert str(wanted / "nextflow.log") in widget.nxf_base_cmd


def test_no_base_dir_uses_the_default(tmp_path):
    widget = DirStub(default=tmp_path / "fallback")

    widget.setup_nxf_dir_cmd()

    assert widget.nxf_base_dir == tmp_path / "fallback"
