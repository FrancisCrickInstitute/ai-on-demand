"""
Translating locally-mounted paths to their location on the remote machine.

An SSH run addresses the same files twice: once through the local mount point
(`/Users/me/nemo/...`, `/Volumes/...`, `Z:\\...`) and once through the path the
remote machine itself uses (`/nemo/stp/...`). Mapping between them by plain
string replacement is unforgiving - if the two prefixes disagree by a single
trailing slash, `/Users/me/nemo/` plus `.hidden` silently becomes `me.hidden`
instead of `me/.hidden`, and nothing complains until a job fails somewhere on
the cluster. A prefix that is not a whole directory (`/Users/me/nemo` against
`/Users/me/nemo2/img.tif`) corrupts the path just as quietly.

`to_remote_path` maps component-wise instead, so trailing slashes do not matter
on either side, only whole directories match, and the result is always a POSIX
path - which is what the remote is, whatever the local machine runs.

Both prefixes must be absolute. `~` is deliberately not expanded: on the remote
side it would mean the *remote* home directory, which this machine cannot know,
and quietly substituting the local one would point the run at a filesystem that
only exists here. Nothing in this module touches the filesystem.
"""

from pathlib import PurePath, PurePosixPath


class RemotePathError(ValueError):
    """
    A local path could not be expressed as a path on the remote machine.

    Raised with a message that already explains what to do about it, so callers
    can surface it as-is rather than wrapping it in their own wording.
    """


def _reject_tilde(value, what: str, extra: str = "") -> None:
    if str(value).startswith("~"):
        raise RemotePathError(
            f"The {what} '{value}' starts with '~', which is not expanded. "
            f"Give the full absolute path instead.{extra}"
        )


def to_remote_path(path, mounted_prefix, remote_prefix) -> str:
    """
    Where `path`, reached locally through `mounted_prefix`, lives on the remote.

    Trailing slashes on either prefix are irrelevant. Raises `RemotePathError`
    if `path` is not inside `mounted_prefix`, since there is then no remote
    location to name.
    """
    _reject_tilde(path, "path")
    _reject_tilde(mounted_prefix, "mounted path prefix")
    local, prefix = PurePath(path), PurePath(mounted_prefix)
    try:
        relative = local.relative_to(prefix)
    except ValueError:
        raise RemotePathError(
            f"'{path}' is not inside the mounted path prefix '{mounted_prefix}', "
            "so it has no location on the remote machine. Either the remote "
            "drive is not mounted where the plugin expects, or the file is not "
            "on it at all."
        ) from None
    # The remote is POSIX whatever this machine is, so rebuild rather than
    # concatenate: only the components of the local path carry over.
    return str(PurePosixPath(str(remote_prefix)).joinpath(*relative.parts))


def check_prefixes(mounted_prefix, remote_prefix) -> None:
    """
    Raise `RemotePathError` if the two path prefixes cannot map onto each other.
    """
    _reject_tilde(
        remote_prefix,
        "remote path prefix",
        extra=" Note that '~' would be your home directory on this machine, "
        "not on the remote one.",
    )
    if not PurePosixPath(remote_prefix).is_absolute():
        raise RemotePathError(
            f"The remote path prefix '{remote_prefix}' is not an absolute path. "
            "It should be the full path the remote machine uses, e.g. "
            "'/nemo/stp/'."
        )
    _reject_tilde(mounted_prefix, "mounted path prefix")
    if not PurePath(mounted_prefix).is_absolute():
        raise RemotePathError(
            f"The mounted path prefix '{mounted_prefix}' is not an absolute "
            "path. It should be where the remote drive is mounted on this "
            "machine, e.g. '/Volumes/'."
        )
