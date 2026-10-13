"""Serialize an experiment without changing its training recipe."""
import fcntl
import os
from pathlib import Path


def acquire_lock(path, inherited_env=None, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inherited = os.environ.get(inherited_env) if inherited_env else None
    if inherited is not None:
        fd = int(inherited)
        actual, expected = os.fstat(fd), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError('Inherited experiment lock does not match the job')
        handle = os.fdopen(os.dup(fd), 'a+')
    else:
        handle = path.open('a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BaseException:
        handle.close()
        raise
    return handle
