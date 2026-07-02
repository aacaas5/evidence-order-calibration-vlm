import io
import json
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_250.json")
OUT = Path("data/gqa/scaled_images")
REPORT = Path("data/gqa/manifests/gqa_scaled_250_download_report.json")
ARCHIVE_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"

OUT.mkdir(parents=True, exist_ok=True)
samples = json.loads(MANIFEST.read_text(encoding="utf-8"))


class RemoteRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        response = self.session.head(url, timeout=60, allow_redirects=True)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.size)
        return self.position

    def read(self, count=-1):
        if count is None or count < 0:
            count = self.size - self.position
        if count == 0 or self.position >= self.size:
            return b""
        end = min(self.position + count, self.size) - 1
        start = self.position
        error = None
        for attempt in range(1, 7):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=120,
                )
                response.raise_for_status()
                content = response.content
                expected = end - start + 1
                if response.status_code != 206 or len(content) != expected:
                    raise RuntimeError(
                        f"Invalid range response status={response.status_code} "
                        f"bytes={len(content)} expected={expected}"
                    )
                self.position += len(content)
                return content
            except Exception as exc:
                error = exc
                time.sleep(min(attempt * 3, 15))
        raise RuntimeError(f"Range request {start}-{end} failed: {error}")


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def write_report(results):
    valid_ids = {
        str(sample["image_id"])
        for sample in samples
        if valid_image(OUT / f'{sample["image_id"]}.jpg')
    }
    missing = [
        str(sample["image_id"])
        for sample in samples
        if str(sample["image_id"]) not in valid_ids
    ]
    payload = {
        "source": ARCHIVE_URL,
        "method": "selective_http_range_extraction",
        "requested": len(samples),
        "valid_images": len(valid_ids),
        "missing": len(missing),
        "missing_ids": missing,
        "results": results,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def fetch_range(session, start, end):
    error = None
    for attempt in range(1, 9):
        try:
            response = session.get(
                ARCHIVE_URL,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=120,
            )
            response.raise_for_status()
            expected = end - start + 1
            if response.status_code != 206 or len(response.content) != expected:
                raise RuntimeError(
                    f"Invalid range status={response.status_code} "
                    f"bytes={len(response.content)} expected={expected}"
                )
            return response.content
        except Exception as exc:
            error = exc
            time.sleep(min(attempt * 2, 12))
    raise RuntimeError(f"Range {start}-{end} failed: {error}")


def extract_member(sample, info):
    question_id = str(sample["question_id"])
    image_id = str(sample["image_id"])
    destination = OUT / f"{image_id}.jpg"
    try:
        if valid_image(destination):
            status = "existing"
        else:
            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0"})
            header = fetch_range(session, info.header_offset, info.header_offset + 29)
            fields = struct.unpack("<4s5H3L2H", header)
            if fields[0] != b"PK\x03\x04":
                raise RuntimeError("Invalid local ZIP header")
            data_start = info.header_offset + 30 + fields[-2] + fields[-1]
            compressed = fetch_range(
                session, data_start, data_start + info.compress_size - 1
            )
            if info.compress_type == zipfile.ZIP_STORED:
                content = compressed
            elif info.compress_type == zipfile.ZIP_DEFLATED:
                content = zlib.decompress(compressed, -15)
            else:
                raise RuntimeError(f"Unsupported ZIP compression: {info.compress_type}")
            if len(content) != info.file_size:
                raise RuntimeError(
                    f"Size mismatch: {len(content)} != {info.file_size}"
                )
            if (zlib.crc32(content) & 0xFFFFFFFF) != info.CRC:
                raise RuntimeError("ZIP CRC mismatch")
            destination.write_bytes(content)
            if not valid_image(destination):
                raise RuntimeError("PIL validation failed")
            status = "downloaded"
        return {
            "question_id": question_id,
            "image_id": image_id,
            "status": status,
            "bytes": destination.stat().st_size,
        }
    except Exception as exc:
        if destination.exists() and not valid_image(destination):
            destination.unlink()
        return {
            "question_id": question_id,
            "image_id": image_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


print(f"P9B requested images: {len(samples)}")
remote = RemoteRangeFile(ARCHIVE_URL)
with zipfile.ZipFile(remote) as archive:
    entries = {
        Path(info.filename).stem: info
        for info in archive.infolist()
        if not info.is_dir() and info.filename.lower().endswith(".jpg")
    }
    print(f"Official archive entries: {len(entries)}")
    jobs = []
    for sample in samples:
        image_id = str(sample["image_id"])
        info = entries.get(image_id)
        if info is None:
            raise KeyError(f"{image_id}.jpg not found in official archive")
        jobs.append((sample, info))

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(extract_member, sample, info): str(sample["image_id"])
            for sample, info in jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(samples)}] {result['image_id']} "
                f"{result['status']}"
                + (f": {result['error']}" if result["status"] == "failed" else "")
            )
            write_report(results)

report = write_report(results)
print("\nP9B SCALED IMAGE DOWNLOAD")
print("Requested:", report["requested"])
print("Valid images:", report["valid_images"])
print("Missing:", report["missing"])
print("Images:", OUT)
print("Report:", REPORT)
if report["missing"]:
    raise RuntimeError(f'P9B incomplete: {report["missing"]} images missing')
print("P9B COMPLETE")
