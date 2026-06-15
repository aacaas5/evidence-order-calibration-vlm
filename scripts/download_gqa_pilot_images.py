import io
import json
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/gqa/manifests/gqa_evidence_pilot_50.json"
OUT_DIR = ROOT / "data/gqa/pilot_images"
REPORT = ROOT / "results/raw/gqa_pilot_image_download_report.json"
SOURCE_URLS = (
    "https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
)


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def valid_bytes(content):
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def make_session():
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.headers["User-Agent"] = "gqa-pilot-image-downloader/1.0"
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_image(session, image_id, destination):
    errors = []
    for template in SOURCE_URLS:
        url = template.format(image_id=image_id)
        try:
            response = session.get(url, timeout=(10, 45))
            response.raise_for_status()
            if not valid_bytes(response.content):
                raise ValueError("response is not a valid image")
            temporary = Path(str(destination) + ".part")
            temporary.write_bytes(response.content)
            temporary.replace(destination)
            return url
        except (OSError, requests.RequestException, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
    image_ids = list(dict.fromkeys(str(sample["image_id"]) for sample in samples))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    session = make_session()
    downloaded = 0
    existing = 0
    failed = []

    for index, image_id in enumerate(image_ids, start=1):
        destination = OUT_DIR / f"{image_id}.jpg"
        if destination.exists() and valid_image(destination):
            existing += 1
            print(f"[{index}/{len(image_ids)}] {image_id}: already existing")
            continue

        try:
            source = download_image(session, image_id, destination)
            downloaded += 1
            print(f"[{index}/{len(image_ids)}] {image_id}: downloaded from {source}")
        except RuntimeError as exc:
            failed.append({"image_id": image_id, "error": str(exc)})
            print(f"[{index}/{len(image_ids)}] {image_id}: FAILED")

    report = {
        "requested": len(image_ids),
        "downloaded": downloaded,
        "already_existing": existing,
        "failed": len(failed),
        "failures": failed,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummary")
    for key in ("requested", "downloaded", "already_existing", "failed"):
        print(f"{key}: {report[key]}")
    print(f"failure report: {REPORT}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()