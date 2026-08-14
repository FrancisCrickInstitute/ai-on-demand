"""
Tests for OpenSSH-equivalent host key verification.

All offline and GUI-free: every test drives a known_hosts file in tmp_path via
an ssh_config that points `UserKnownHostsFile` at it, which also exercises the
ssh_config resolution itself.
"""

import paramiko
import pytest
from paramiko.hostkeys import HostKeyEntry, HostKeys

from aiod_napari.inference.ssh_hostkeys import (
    HostKeyError,
    KnownHostsPolicy,
    assert_host_known,
    keyscan_type,
    known_hosts_paths,
    lookup,
)

HOST = "login.nemo.thecrick.org"


def new_key():
    # Ed25519Key has no generate() in paramiko, and ECDSA is fast enough to
    # make a fresh key per test.
    return paramiko.ECDSAKey.generate()


@pytest.fixture
def ssh_dir(tmp_path):
    """
    A (known_hosts, ssh_config) pair, isolated from the developer's own ~/.ssh.
    """
    known_hosts = tmp_path / "known_hosts"
    known_hosts.touch()
    config = tmp_path / "config"
    config.write_text(
        "Host *\n"
        f"  UserKnownHostsFile {known_hosts}\n"
        "  GlobalKnownHostsFile /dev/null\n"
    )
    return known_hosts, config


def write_entries(known_hosts, entries: list[tuple[str, paramiko.PKey]]):
    known_hosts.write_text(
        "".join(HostKeyEntry([name], key).to_line() for name, key in entries)
    )


def test_all_keys_recorded_for_one_host_are_accepted(ssh_dir):
    """
    The load-balancer case: one name, several nodes, a different key each.

    Paramiko's own check only ever compares against the first entry of a given
    type, so this is the behaviour that motivates the whole module.
    """
    known_hosts, config = ssh_dir
    keys = [new_key() for _ in range(3)]
    write_entries(known_hosts, [(HOST, key) for key in keys])

    policy = KnownHostsPolicy(HOST, ssh_config_path=config)
    for key in keys:
        policy.missing_host_key(None, HOST, key)  # must not raise

    # Paramiko, for contrast, accepts only the first of the three.
    paramiko_check = HostKeys(str(known_hosts))
    assert paramiko_check.check(HOST, keys[0])
    assert not paramiko_check.check(HOST, keys[1])
    assert not paramiko_check.check(HOST, keys[2])


def test_hashed_entry_matches(ssh_dir):
    known_hosts, config = ssh_dir
    key = new_key()
    hashed = HostKeys.hash_host(HOST)
    known_hosts.write_text(f"{hashed} {key.get_name()} {key.get_base64()}\n")

    KnownHostsPolicy(HOST, ssh_config_path=config).missing_host_key(None, HOST, key)


def test_wildcard_entry_matches(ssh_dir):
    known_hosts, config = ssh_dir
    key = new_key()
    write_entries(known_hosts, [("ga*", key)])

    KnownHostsPolicy("ga134", ssh_config_path=config).missing_host_key(
        None, "ga134", key
    )
    with pytest.raises(HostKeyError):
        KnownHostsPolicy("cn092", ssh_config_path=config).missing_host_key(
            None, "cn092", key
        )


def test_comma_separated_names_match(ssh_dir):
    known_hosts, config = ssh_dir
    key = new_key()
    known_hosts.write_text(f"{HOST},10.28.4.18 {key.get_name()} {key.get_base64()}\n")

    for name in (HOST, "10.28.4.18"):
        KnownHostsPolicy(name, ssh_config_path=config).missing_host_key(None, name, key)


def test_unknown_host_says_how_to_trust_it(ssh_dir):
    known_hosts, config = ssh_dir
    key = new_key()

    policy = KnownHostsPolicy(HOST, ssh_config_path=config)
    with pytest.raises(HostKeyError) as excinfo:
        policy.missing_host_key(None, HOST, key)

    message = str(excinfo.value)
    assert "not in your known_hosts" in message
    assert key.fingerprint in message
    assert f"ssh-keyscan -t ecdsa {HOST} >> {known_hosts}" in message


def test_wrong_key_names_the_offending_file_and_line(ssh_dir):
    known_hosts, config = ssh_dir
    recorded, offered = new_key(), new_key()
    write_entries(known_hosts, [("other.host", recorded), (HOST, recorded)])

    policy = KnownHostsPolicy(HOST, ssh_config_path=config)
    with pytest.raises(HostKeyError) as excinfo:
        policy.missing_host_key(None, HOST, offered)

    message = str(excinfo.value)
    assert f"{known_hosts}:2" in message
    assert recorded.fingerprint in message
    assert offered.fingerprint in message
    assert f"ssh-keygen -R {HOST}" in message
    # The unrelated host's line is not implicated
    assert f"{known_hosts}:1" not in message


def test_dev_null_known_hosts_is_reported_as_unverifiable(tmp_path):
    config = tmp_path / "config"
    config.write_text(
        "Host *\n  UserKnownHostsFile /dev/null\n  GlobalKnownHostsFile /dev/null\n"
    )

    assert known_hosts_paths(HOST, ssh_config_path=config) == []
    with pytest.raises(ValueError, match="/dev/null"):
        assert_host_known(HOST, ssh_config_path=config)


def test_unparseable_lines_are_skipped(ssh_dir):
    known_hosts, config = ssh_dir
    key = new_key()
    known_hosts.write_text(
        "# a comment\n"
        "\n"
        f"@revoked {HOST} {key.get_name()} {key.get_base64()}\n"
        "not-a-known-hosts-line\n"
        f"{HOST} {key.get_name()} !!!not-base64!!!\n"
        + HostKeyEntry([HOST], key).to_line()
    )

    entries = lookup(HOST, ssh_config_path=config)
    assert len(entries) == 1
    assert entries[0].lineno == 6


def test_assert_host_known(ssh_dir):
    known_hosts, config = ssh_dir
    write_entries(known_hosts, [(HOST, new_key())])

    assert_host_known(HOST, ssh_config_path=config)
    with pytest.raises(ValueError, match="ssh-keyscan"):
        assert_host_known("no.such.host", ssh_config_path=config)


def test_recorded_key_types_are_preferred(ssh_dir):
    """
    An RSA-only entry must make the client ask for RSA, including the SHA-2
    signature variants a modern server actually offers.
    """
    known_hosts, config = ssh_dir
    write_entries(known_hosts, [(HOST, paramiko.RSAKey.generate(2048))])

    policy = KnownHostsPolicy(HOST, ssh_config_path=config)
    assert policy.known_key_types == ["rsa-sha2-512", "rsa-sha2-256", "ssh-rsa"]


def test_no_ssh_config_falls_back_to_defaults(tmp_path):
    missing = tmp_path / "does-not-exist"
    # Should not raise, and should not consult the missing config
    known_hosts_paths(HOST, ssh_config_path=missing)


@pytest.mark.parametrize(
    ("key_name", "expected"),
    [
        ("ssh-ed25519", "ed25519"),
        ("ssh-rsa", "rsa"),
        ("rsa-sha2-512", "rsa"),
        ("ecdsa-sha2-nistp256", "ecdsa"),
        ("ecdsa-sha2-nistp521", "ecdsa"),
    ],
)
def test_keyscan_type(key_name, expected):
    assert keyscan_type(key_name) == expected
