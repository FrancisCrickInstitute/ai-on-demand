"""
OpenSSH-equivalent host key verification for the SSH execution path.

Paramiko's built-in verification is not equivalent to OpenSSH's when a host has
more than one key of the same type recorded against it: ``HostKeys.check()``
only ever compares against the *first* matching entry, so a load-balanced name
that resolves to several machines with different host keys (e.g. an HPC login
alias) fails for every node but one. Paramiko also reads only
``~/.ssh/known_hosts``, ignoring any ``UserKnownHostsFile`` or
``GlobalKnownHostsFile`` set in ``~/.ssh/config``.

`KnownHostsPolicy` replaces that check with OpenSSH's own semantics: the offered
key is accepted if it matches *any* entry recorded for the host. Unknown hosts
are still refused - the policy just says what to run to trust one.

Nothing here reads ``HostName``, ``User``, ``IdentityFile``, ``Port`` or
``ProxyJump`` from ``~/.ssh/config``; only the known-hosts file locations are
taken from it, so the values shown in the GUI remain the single source of truth
for what is being connected to.
"""

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import paramiko
from paramiko.hostkeys import HostKeyEntry, HostKeys, InvalidHostKey
from paramiko.pkey import PKey
from paramiko.ssh_exception import SSHException
from paramiko.util import constant_time_bytes_eq

SSH_CONFIG_PATH = Path("~/.ssh/config")
DEFAULT_USER_KNOWN_HOSTS = Path("~/.ssh/known_hosts")
DEFAULT_GLOBAL_KNOWN_HOSTS = Path("/etc/ssh/ssh_known_hosts")

# Key names as they appear in known_hosts, mapped to the argument that
# `ssh-keyscan -t` expects.
_KEYSCAN_TYPES = {
    "ssh-ed25519": "ed25519",
    "ssh-rsa": "rsa",
    "rsa-sha2-256": "rsa",
    "rsa-sha2-512": "rsa",
    "ssh-dss": "dsa",
}


class HostKeyError(SSHException):
    """
    A host key could not be verified against known_hosts.

    Raised with a message that already explains what to do about it, so callers
    can surface it as-is rather than wrapping it in their own wording.
    """


@dataclass(frozen=True)
class KnownHost:
    """A single usable line of a known_hosts file, with its origin."""

    path: Path
    lineno: int
    hostnames: list[str]
    key: PKey

    def describe(self) -> str:
        return f"{self.path}:{self.lineno} {self.key.get_name()} {self.key.fingerprint}"


def keyscan_type(key_name: str) -> str:
    """
    The `ssh-keyscan -t` argument for a key named as known_hosts names it.
    """
    if key_name.startswith("ecdsa-"):
        return "ecdsa"
    return _KEYSCAN_TYPES.get(key_name, key_name)


def _config_known_hosts(
    host: str, ssh_config_path: Path
) -> tuple[list[str], list[str]]:
    """
    The `UserKnownHostsFile` and `GlobalKnownHostsFile` entries that apply to
    `host`, or empty lists if there is no readable ssh_config.

    Each option may name several files, separated by whitespace.
    """
    ssh_config_path = Path(ssh_config_path).expanduser()
    if not ssh_config_path.is_file():
        return [], []
    try:
        config = paramiko.SSHConfig.from_path(str(ssh_config_path))
    except Exception:  # noqa: BLE001 - a broken config shouldn't break the run
        return [], []
    options = config.lookup(host)
    return (
        options.get("userknownhostsfile", "").split(),
        options.get("globalknownhostsfile", "").split(),
    )


def user_known_hosts_path(
    host: str, ssh_config_path: Path = SSH_CONFIG_PATH
) -> Path | None:
    """
    Where a new key for `host` should be written, whether or not it exists yet.

    Returns None when the user has deliberately discarded host keys for this
    host with `UserKnownHostsFile /dev/null`, in which case there is nowhere to
    suggest writing to.
    """
    user_files, _ = _config_known_hosts(host, ssh_config_path)
    for candidate in user_files or [str(DEFAULT_USER_KNOWN_HOSTS)]:
        path = Path(candidate).expanduser()
        if path != Path("/dev/null"):
            return path
    return None


def known_hosts_paths(host: str, ssh_config_path: Path = SSH_CONFIG_PATH) -> list[Path]:
    """
    The existing known_hosts files OpenSSH would consult for `host`.

    `UserKnownHostsFile`/`GlobalKnownHostsFile` from ssh_config take precedence
    over the defaults, `/dev/null` is skipped, and user files are searched
    before global ones.
    """
    user_files, global_files = _config_known_hosts(host, ssh_config_path)
    candidates = (user_files or [str(DEFAULT_USER_KNOWN_HOSTS)]) + (
        global_files or [str(DEFAULT_GLOBAL_KNOWN_HOSTS)]
    )
    paths = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path == Path("/dev/null") or path in paths or not path.is_file():
            continue
        paths.append(path)
    return paths


def load_entries(paths: list[Path]) -> list[KnownHost]:
    """
    Parse every usable line of the given known_hosts files.

    Lines that paramiko cannot parse are skipped, as OpenSSH does. That
    includes `@cert-authority` and `@revoked` markers, which are not supported
    here - a `@revoked` key is treated as simply unknown rather than as an
    explicit refusal.
    """
    entries = []
    for path in paths:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            try:
                entry = HostKeyEntry.from_line(line, lineno)
            except InvalidHostKey:
                continue
            if entry is None or entry.key is None:
                continue
            entries.append(
                KnownHost(
                    path=path,
                    lineno=lineno,
                    hostnames=list(entry.hostnames),
                    key=entry.key,
                )
            )
    return entries


def _hostname_matches(host: str, pattern: str) -> bool:
    if pattern.startswith("|1|"):
        # Hashed entry: re-hash the hostname with the entry's own salt.
        return constant_time_bytes_eq(HostKeys.hash_host(host, pattern), pattern)
    if pattern == host:
        return True
    # OpenSSH allows `*` and `?` patterns, which paramiko itself does not match.
    return ("*" in pattern or "?" in pattern) and fnmatch(host, pattern)


def matching_keys(host: str, entries: list[KnownHost]) -> list[KnownHost]:
    """
    Every entry recorded against `host`, in the order the files list them.
    """
    return [
        entry
        for entry in entries
        if any(_hostname_matches(host, name) for name in entry.hostnames)
    ]


def lookup(host: str, ssh_config_path: Path = SSH_CONFIG_PATH) -> list[KnownHost]:
    """
    Every known_hosts entry recorded against `host`.
    """
    return matching_keys(host, load_entries(known_hosts_paths(host, ssh_config_path)))


def _dev_null_hint() -> str:
    return (
        "Your ssh_config sets 'UserKnownHostsFile /dev/null' for this host, "
        "so host keys are discarded and cannot be verified. Point that at a "
        "real file to connect from the plugin."
    )


def shell_quote(value) -> str:
    """
    Quote a value for a command line the user is going to paste into a shell.

    Double quotes rather than the POSIX single quotes `shlex.quote` would use:
    these commands are read by people on macOS, Linux and Windows, and double
    quotes are the only form `sh`, PowerShell and `cmd` all understand.
    """
    text = str(value)
    if text and all(c.isalnum() or c in "._/@:+-\\" for c in text):
        return text
    return '"' + text.replace('"', '\\"') + '"'


def ssh_target(user: str | None, host: str) -> str:
    """
    A host as it would be typed at a shell, username included when known.

    The remote username need not match the local one, so any advice of the form
    `ssh <host> ...` has to carry it or it will not run as the right user.
    """
    return f"{user}@{host}" if user else host


def _goes_through(host: str, via: str | None) -> bool:
    """
    Whether `host` is reached through `via`, which is not a jump host to itself.

    `via` may carry a 'user@' prefix, which is not part of the host name.
    """
    return bool(via) and via.rpartition("@")[2] != host


def route_note(host: str, via: str | None) -> str:
    """
    Why the obvious command will not work for a host behind a jump host.

    Empty when there is no jump host to explain.
    """
    if not _goes_through(host, via):
        return ""
    return (
        f"ssh-keyscan cannot see '{host}' directly: it does its own lookup and "
        f"does not read ssh_config, so it cannot go through '{via}'. The key has "
        "to be fetched over the jump host instead."
    )


def record_command(
    host: str,
    target: Path,
    key_type: str | None = None,
    via: str | None = None,
    identity: str | None = None,
) -> str:
    """
    The one command that appends `host`'s key to `target`, and nothing else.

    Routed over `via` when the host is only addressable through it - which is
    what the target node of an HPC run is - so the key arrives over a
    connection whose own host key is already verified.
    """
    flag = f"-t {key_type} " if key_type else ""
    redirect = f">> {shell_quote(target)}"
    if _goes_through(host, via):
        key_opt = f"-i {shell_quote(identity)} " if identity else ""
        return f"ssh {key_opt}{via} ssh-keyscan {flag}{host} {redirect}"
    return f"ssh-keyscan {flag}{host} {redirect}"


def record_hint(
    host: str,
    target: Path | None,
    key_type: str | None = None,
    via: str | None = None,
    identity: str | None = None,
) -> str:
    """
    Full advice for trusting a host that is not on record yet.

    Ends with an ssh_config-aware fallback, which only makes sense while there
    is no conflicting key on file - see `record_command` for the bare command
    to use when there is.
    """
    if target is None:
        return _dev_null_hint()
    note = route_note(host, via)
    command = record_command(host, target, key_type, via, identity)
    return (
        (f"{note}\n" if note else "")
        + f"If you trust it, record its host key with:\n    {command}\n"
        + "Or let ssh do it, which reads your ssh_config and shows you the "
        + "fingerprint to confirm:\n    "
        + f"ssh -o StrictHostKeyChecking=ask {host} true"
    )


def assert_host_known(
    host: str,
    ssh_config_path: Path = SSH_CONFIG_PATH,
    via: str | None = None,
    identity: str | None = None,
) -> None:
    """
    Raise `ValueError` if `host` has no host key on record.

    A cheap offline pre-flight, so an unknown host is reported before a pipeline
    starts rather than when the connection is attempted. `via` is the jump host
    `host` is reached through and `identity` the key file used to get there, so
    that the advice given can actually be run.
    """
    if lookup(host, ssh_config_path):
        return
    target = user_known_hosts_path(host, ssh_config_path)
    raise ValueError(
        f"The host '{host}' is not in your known_hosts files, so its identity "
        "cannot be verified.\n"
        f"{record_hint(host, target, via=via, identity=identity)}"
    )


class KnownHostsPolicy(paramiko.MissingHostKeyPolicy):
    """
    Verifies the server's host key against known_hosts the way OpenSSH does.

    The entries for the host are read once, at construction. Paramiko calls
    `missing_host_key` after the key exchange but before authentication, so
    refusing here means no credentials are ever sent to an unverified server.

    Pass `transport_factory` to `SSHClient.connect` alongside this policy, and
    do not load any host keys into the client - paramiko only consults the
    policy for hosts it does not know itself.
    """

    def __init__(
        self,
        host: str,
        ssh_config_path: Path = SSH_CONFIG_PATH,
        via: str | None = None,
        identity: str | None = None,
    ):
        self.host = host
        self.via = via
        self.identity = identity
        self.searched = known_hosts_paths(host, ssh_config_path)
        self.entries = load_entries(self.searched)
        self.matches = matching_keys(host, self.entries)
        self._ssh_config_path = ssh_config_path

    @property
    def known_key_types(self) -> list[str]:
        """
        The key types on record for this host, most-preferred first.

        `ssh-rsa` entries also cover the SHA-2 signature variants, which is what
        a modern server will actually offer for an RSA host key.
        """
        types = []
        for entry in self.matches:
            name = entry.key.get_name()
            expanded = (
                ["rsa-sha2-512", "rsa-sha2-256", "ssh-rsa"]
                if name == "ssh-rsa"
                else [name]
            )
            types += [t for t in expanded if t not in types]
        return types

    def transport_factory(self, sock, **kwargs) -> paramiko.Transport:
        """
        Build the transport, preferring host key types we have on record.

        Paramiko normally does this itself, but only for hosts it verifies
        itself; without it a host recorded as `ssh-rsa`-only would be offered an
        ed25519 key and reported as a mismatch.
        """
        transport = paramiko.Transport(sock, **kwargs)
        known = self.known_key_types
        if known:
            options = transport.get_security_options()
            preferred = [k for k in known if k in options.key_types]
            options.key_types = preferred + [
                k for k in options.key_types if k not in preferred
            ]
        return transport

    def missing_host_key(self, client, hostname: str, key: PKey) -> None:
        matches = (
            self.matches
            if hostname == self.host
            else matching_keys(hostname, self.entries)
        )
        offered = key.asbytes()
        if any(
            constant_time_bytes_eq(offered, entry.key.asbytes()) for entry in matches
        ):
            return
        raise HostKeyError(
            self._mismatch_message(hostname, key, matches)
            if matches
            else self._unknown_message(hostname, key)
        )

    def _offered(self, key: PKey) -> str:
        return f"a {key.get_name()} key with fingerprint {key.fingerprint}"

    def _unknown_message(self, hostname: str, key: PKey) -> str:
        searched = ", ".join(str(p) for p in self.searched) or "none found"
        target = user_known_hosts_path(hostname, self._ssh_config_path)
        message = (
            f"The host '{hostname}' is not in your known_hosts files, so its "
            f"identity cannot be verified. It offered {self._offered(key)}.\n"
            f"Files searched: {searched}."
        )
        hint = record_hint(
            hostname,
            target,
            keyscan_type(key.get_name()),
            self.via,
            self.identity,
        )
        return f"{message}\nVerify that fingerprint first.\n{hint}"

    def _mismatch_message(
        self, hostname: str, key: PKey, matches: list[KnownHost]
    ) -> str:
        recorded = "\n".join(f"    {entry.describe()}" for entry in matches)
        target = user_known_hosts_path(hostname, self._ssh_config_path)
        # Only the bare command here: with a conflicting key already on file,
        # ssh itself refuses to connect, so record_hint's ssh fallback would
        # be advice that cannot work
        if target is None:
            keep_both = f"    {_dev_null_hint()}"
        else:
            note = route_note(hostname, self.via)
            command = record_command(
                hostname,
                target,
                keyscan_type(key.get_name()),
                self.via,
                self.identity,
            )
            keep_both = (f"{note}\n" if note else "") + f"    {command}"
        return (
            f"Host key verification failed for '{hostname}'. It offered "
            f"{self._offered(key)}, which matches none of the "
            f"{len(matches)} key(s) on record:\n{recorded}\n"
            "Either the host key has changed (which can mean a "
            "man-in-the-middle), or this name resolves to several machines with "
            "different host keys. Verify the fingerprint above by another "
            "route, then either drop the stale entries with\n"
            f"    ssh-keygen -R {hostname}\n"
            "or, if the other machines are still valid, add this key "
            f"alongside them:\n{keep_both}"
        )
