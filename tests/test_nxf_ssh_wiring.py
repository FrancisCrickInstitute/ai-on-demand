"""
The SSH pre-flight has to hand the host-key advice enough context to be usable.

A target node is reached through the jump host, so `assert_host_known` needs the
jump host and the identity file or the command it prints cannot be run. These
tests pin the wiring itself: without them the `via=`/`identity=` arguments could
be dropped and every other test would still pass.
"""

from pathlib import Path

import pytest

import aiod_napari.inference.nxf as nxf
from aiod_napari.inference.nxf import NxfWidget


class Field:
    """The bit of QLineEdit that check_ssh_options actually uses."""

    def __init__(self, value=""):
        self._value = value

    def text(self):
        return self._value


class SshOptionsStub:
    """The SSH-checking half of NxfWidget, without needing a viewer."""

    check_ssh_options = NxfWidget.check_ssh_options
    _jump_host = NxfWidget._jump_host
    _to_remote = NxfWidget._to_remote

    def __init__(self, **overrides):
        self.hostname = Field("login.example.org")
        self.target_node = Field("cn093")
        self.username = Field("clusteruser")
        self.remote_path_prefix = Field("/remote/data")
        self.mounted_path_prefix = Field("/Volumes/data")
        self.command_prepend = Field("module load Nextflow")
        self.ssh_key_path = "/keys/id_rsa"
        self.nxf_base_dir = Path("/Volumes/data/run")
        for name, field in overrides.items():
            setattr(self, name, field)


@pytest.fixture
def recorded_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nxf, "assert_host_known", lambda host, **kwargs: calls.append((host, kwargs))
    )
    return calls


def test_target_node_is_checked_with_the_jump_host_and_identity(recorded_calls):
    SshOptionsStub().check_ssh_options()

    assert recorded_calls == [
        ("login.example.org", {}),
        (
            "cn093",
            {"via": "clusteruser@login.example.org", "identity": "/keys/id_rsa"},
        ),
    ]


def test_the_jump_host_itself_is_checked_without_a_route(recorded_calls):
    """
    Nothing stands in front of the jump host, so no `via` for it.
    """
    SshOptionsStub(target_node=Field("")).check_ssh_options()

    assert recorded_calls == [("login.example.org", {})]


def test_jump_host_carries_the_username_when_it_differs_from_the_local_one():
    assert SshOptionsStub()._jump_host() == "clusteruser@login.example.org"


def test_jump_host_without_a_username_is_just_the_host():
    assert SshOptionsStub(username=Field(""))._jump_host() == "login.example.org"


def test_no_identity_is_passed_as_none_rather_than_an_empty_string(recorded_calls):
    """
    An empty GUI field must not become `-i ""` in the advice.
    """
    stub = SshOptionsStub()
    stub.ssh_key_path = ""

    with pytest.raises(ValueError, match="SSH Key"):
        stub.check_ssh_options()


def test_prefix_mismatch_is_still_caught_after_the_host_checks(recorded_calls):
    stub = SshOptionsStub(mounted_path_prefix=Field("/Volumes/other"))

    with pytest.raises(ValueError, match="does not start with"):
        stub.check_ssh_options()
