"""Leer archivos sueltos de un zip remoto con requests HTTP por rango (sin bajar el zip entero)."""
import bz2
import io
import struct
import time
import urllib.request
import zipfile
import zlib


class HttpFile(io.RawIOBase):
    """Archivo remoto de solo lectura, con seek, sobre HTTP Range."""

    def __init__(self, url):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()  # sigue redirects (HF -> CDN)
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        data = get_range(self.url, self.pos, min(self.pos + n, self.size) - 1)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def get_range(url, start, end, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def index(url):
    """Devuelve {nombre: (header_offset, compress_size, compress_type)} leyendo solo el directorio central."""
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 22))
    return {i.filename: (i.header_offset, i.compress_size, i.compress_type) for i in z.infolist() if not i.is_dir()}


def fetch(url, header_offset, compress_size, compress_type):
    """Baja un único miembro del zip (header local + datos) y lo devuelve descomprimido."""
    head = get_range(url, header_offset, header_offset + 29)
    sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", head)
    assert sig == 0x04034B50, "header local inválido"
    start = header_offset + 30 + name_len + extra_len
    data = get_range(url, start, start + compress_size - 1)
    assert len(data) == compress_size
    if compress_type == zipfile.ZIP_STORED:
        return data
    if compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    if compress_type == zipfile.ZIP_BZIP2:  # FMA usa bzip2
        return bz2.decompress(data)
    raise ValueError(f"compresión no soportada: {compress_type}")
