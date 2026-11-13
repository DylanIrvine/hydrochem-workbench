# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.
#
# Taken from HydroSTITCH, with the key names changed.

# core/usage.py
# usage counter. Uses Upstash Redis if a url and token are given, otherwise a
# local json file (.run_count.json), which resets when Streamlit Cloud redeploys
# default redis key is "hydrochem:runs"

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".run_count.json"
DEFAULT_REDIS_KEY = "hydrochem:runs"


# %% redis backend
def _redis_incr(url, token, key, start=0):
    from upstash_redis import Redis

    client = Redis(url=url, token=token)
    if start:
        client.set(key, int(start), nx=True)
    return int(client.incr(key))


def _redis_get(url, token, key):
    from upstash_redis import Redis

    value = Redis(url=url, token=token).get(key)
    return int(value) if value is not None else 0


# %% file backend
def _resolve_path(explicit=None):
    if explicit:
        return Path(explicit)
    env = os.environ.get("HYDROCHEM_RUN_COUNT_PATH")
    return Path(env) if env else _DEFAULT_PATH


def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return int(data.get("runs", 0)), dict(data)
    except (FileNotFoundError, ValueError, OSError, TypeError):
        return 0, {}


def _write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".runcount-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# %% main functions
def increment(path=None, start=0, *, redis_url=None, redis_token=None,
              redis_key=DEFAULT_REDIS_KEY):
    # add 1 and return the new total (None if nothing is available)
    if redis_url and redis_token:
        try:
            return _redis_incr(redis_url, redis_token, redis_key or DEFAULT_REDIS_KEY,
                               start)
        except Exception:
            pass   # fall back to the file

    target = _resolve_path(path)
    try:
        current, data = _read(target)
        base = max(current, int(start or 0))
        new = base + 1
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        data["runs"] = new
        data.setdefault("first_iso", now)
        data["last_iso"] = now
        _write_atomic(target, data)
        return new
    except OSError:
        return None


def read_only(path=None, *, redis_url=None, redis_token=None,
              redis_key=DEFAULT_REDIS_KEY):
    # current count, without adding to it
    if redis_url and redis_token:
        try:
            return _redis_get(redis_url, redis_token, redis_key or DEFAULT_REDIS_KEY)
        except Exception:
            pass
    current, _ = _read(_resolve_path(path))
    return current


def status(path=None, *, redis_url=None, redis_token=None,
           redis_key=DEFAULT_REDIS_KEY):
    # which backend is in use ('redis', 'redis-error', 'file' or 'memory') and the count
    if redis_url and redis_token:
        try:
            n = _redis_get(redis_url, redis_token, redis_key or DEFAULT_REDIS_KEY)
            return {'backend': 'redis', 'count': int(n),
                    'detail': redis_key or DEFAULT_REDIS_KEY}
        except Exception as exc:
            return {'backend': 'redis-error', 'count': None,
                    'detail': f'{type(exc).__name__}: {exc}'}
    try:
        current, _ = _read(_resolve_path(path))
        return {'backend': 'file', 'count': int(current),
                'detail': str(_resolve_path(path))}
    except OSError:
        return {'backend': 'memory', 'count': None, 'detail': ''}
