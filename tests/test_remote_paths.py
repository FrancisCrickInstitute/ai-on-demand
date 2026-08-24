"""
Tests for mapping locally-mounted paths onto the remote machine.

The CI matrix runs ubuntu, windows and macos, so the platform-guarded cases
below are genuinely exercised rather than skipped everywhere.
"""

import sys
from pathlib import Path

import pytest

from aiod_napari.inference.remote_paths import (
    RemotePathError,
    check_prefixes,
    to_remote_path,
)

MOUNTED = "/Users/yuq/nemo"
REMOTE = "/nemo/stp/ddt/working/yuq"


@pytest.mark.parametrize("mounted", [MOUNTED, MOUNTED + "/"])
@pytest.mark.parametrize("remote", [REMOTE, REMOTE + "/"])
def test_trailing_slashes_do_not_matter(mounted, remote):
    """
    The regression: a lone trailing slash used to turn '/Users/yuq/nemo/' plus
    '.hidden' into 'yuq.hidden' instead of 'yuq/.hidden'.
    """
    mapped = to_remote_path(f"{MOUNTED}/.hidden/.nextflow/run1", mounted, remote)
    assert mapped == f"{REMOTE}/.hidden/.nextflow/run1"


def test_sibling_directory_is_not_a_prefix_match():
    """
    A plain str.replace would map '/Users/yuq/nemo2' onto the remote too.
    """
    with pytest.raises(RemotePathError, match="not inside the mounted path prefix"):
        to_remote_path("/Users/yuq/nemo2/img.tif", MOUNTED, REMOTE)


def test_path_outside_the_mount_is_rejected():
    with pytest.raises(RemotePathError) as excinfo:
        to_remote_path("/Users/yuq/Downloads/img.tif", MOUNTED, REMOTE)
    message = str(excinfo.value)
    assert "/Users/yuq/Downloads/img.tif" in message
    assert MOUNTED in message


def test_the_mount_point_itself_maps_to_the_remote_root():
    assert to_remote_path(MOUNTED, MOUNTED, REMOTE) == REMOTE


def test_accepts_path_objects_and_returns_a_string():
    mapped = to_remote_path(Path(MOUNTED) / "data" / "img.tif", MOUNTED, REMOTE)
    assert mapped == f"{REMOTE}/data/img.tif"
    assert isinstance(mapped, str)


def test_names_with_dots_and_spaces_survive():
    mapped = to_remote_path(f"{MOUNTED}/my data/.hidden.tif", MOUNTED, REMOTE)
    assert mapped == f"{REMOTE}/my data/.hidden.tif"


# `~` is never expanded - both prefixes must be absolute


def test_tilde_in_the_mounted_prefix_is_refused():
    with pytest.raises(RemotePathError, match="not expanded"):
        check_prefixes("~/nemo/", REMOTE)


def test_tilde_in_the_remote_prefix_is_refused():
    """
    `~` on the remote is the remote's home, which this machine cannot know -
    expanding it locally would silently point at the wrong filesystem.
    """
    with pytest.raises(RemotePathError, match="not on the remote one"):
        check_prefixes(MOUNTED, "~/working/yuq/")


def test_tilde_in_the_mapped_path_is_refused():
    """
    A shared config carrying '~' should say so, not fail as a prefix mismatch.
    """
    with pytest.raises(RemotePathError, match="not expanded"):
        to_remote_path("~/nemo/img.tif", MOUNTED, REMOTE)


def test_tilde_mounted_prefix_is_refused_when_mapping():
    with pytest.raises(RemotePathError, match="not expanded"):
        to_remote_path(f"{MOUNTED}/img.tif", "~/nemo", REMOTE)


def test_relative_remote_prefix_is_rejected():
    with pytest.raises(RemotePathError, match="not an absolute path"):
        check_prefixes(MOUNTED, "nemo/stp/ddt/working/yuq")


def test_relative_mounted_prefix_is_rejected():
    with pytest.raises(RemotePathError, match="not an absolute"):
        check_prefixes("nemo", REMOTE)


@pytest.mark.parametrize("mounted", [MOUNTED, MOUNTED + "/"])
@pytest.mark.parametrize("remote", [REMOTE, REMOTE + "/"])
def test_valid_prefixes_pass(mounted, remote):
    check_prefixes(mounted, remote)


# Cross-platform: the local side follows this machine's flavour, the remote is
# always POSIX. That asymmetry is what makes Windows -> Linux mapping work.


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path flavour")
@pytest.mark.parametrize("mounted", ["Z:\\data", "Z:\\data\\", "Z:/data"])
def test_windows_drive_maps_to_posix_remote(mounted):
    mapped = to_remote_path("Z:\\data\\sub dir\\img.tif", mounted, REMOTE)
    assert mapped == f"{REMOTE}/sub dir/img.tif"
    assert "\\" not in mapped


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path flavour")
def test_windows_unc_mount_maps_to_posix_remote():
    mapped = to_remote_path(
        "\\\\server\\share\\data\\img.tif", "\\\\server\\share\\data", REMOTE
    )
    assert mapped == f"{REMOTE}/img.tif"
    assert "\\" not in mapped


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path flavour")
def test_rootless_mounted_prefix_is_rejected_on_windows():
    """
    '/data' has no drive on Windows, so it cannot be a mount point there.
    """
    with pytest.raises(RemotePathError, match="not an absolute"):
        check_prefixes("/data", REMOTE)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX path flavour")
def test_posix_mount_maps_to_posix_remote():
    mapped = to_remote_path("/Volumes/data/sub dir/img.tif", "/Volumes/data", REMOTE)
    assert mapped == f"{REMOTE}/sub dir/img.tif"
    assert "\\" not in mapped
