#!/usr/bin/env python3
"""List / extract members of a remote zip via HTTP Range requests, without downloading the whole archive.
Stdlib only.  usage:
  remote_zip.py ls  URL [DEPTH]
  remote_zip.py get URL REGEX DEST_DIR [MAX_PER_DIR]
"""
import collections
import http.client
import io
import os
import re
import sys
import time
import urllib.request
import zipfile
import socket

# Try IPv4 before IPv6: urllib connects to each resolved address in turn with the full timeout, so a host whose IPv6
# route is dead (seen here: zenodo.org) stalls for minutes per request, while curl's happy-eyeballs hides it.
_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: sorted(_getaddrinfo(*a, **k), key=lambda r: r[0] != socket.AF_INET)


class HTTPFile(io.RawIOBase):
    def __init__(self, url):
        self.pos = 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        for attempt in range(6):  # a stalled range request otherwise hangs forever
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except (OSError, TimeoutError, http.client.HTTPException) as e:
                if attempt == 5:
                    raise
                print(f"\n  range request failed ({e}); retry {attempt + 1}", file=sys.stderr)
                time.sleep(2 ** attempt)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote(url):
    return zipfile.ZipFile(io.BufferedReader(HTTPFile(url), buffer_size=1 << 20))


def ls(url, depth=3):
    z = open_remote(url)
    agg = collections.defaultdict(lambda: [0, 0])
    for i in z.infolist():
        parts = i.filename.split("/")
        k = "/".join(parts[: min(len(parts) - 1, depth)])
        agg[k][0] += 1
        agg[k][1] += i.compress_size
    for k, (n, s) in sorted(agg.items()):
        print(f"{n:6d} {s / 1e6:8.1f} MB  {k}/")


def get(url, pattern, dest, max_per_dir=0):
    z, pat, dest = open_remote(url), re.compile(pattern), os.path.abspath(dest)
    cnt, total = collections.Counter(), 0
    for i in z.infolist():
        if i.is_dir() or "__MACOSX" in i.filename or not pat.search(i.filename):
            continue
        d = os.path.dirname(i.filename)
        if max_per_dir and cnt[d] >= max_per_dir:
            continue
        out = os.path.abspath(os.path.join(dest, i.filename))
        if not out.startswith(dest + os.sep):
            raise ValueError(f"unsafe path in archive: {i.filename}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == i.file_size:  # resume: already extracted
            cnt[d] += 1
            total += i.file_size
            continue
        with z.open(i) as src, open(out + ".part", "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        os.replace(out + ".part", out)
        cnt[d] += 1
        total += i.file_size
        print(f"\r{sum(cnt.values())} files, {total / 1e6:.0f} MB", end="", flush=True)
    print(f"\nextracted {sum(cnt.values())} files ({total / 1e6:.1f} MB) into {dest}")


if __name__ == "__main__":
    cmd, url = sys.argv[1], sys.argv[2]
    if cmd == "ls":
        ls(url, int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        get(url, sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0)
