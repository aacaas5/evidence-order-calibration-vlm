import zipfile
from pathlib import Path
import fsspec

URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
IMAGE_ID = "2379780"

out = Path("data/gqa/pilot_images")
out.mkdir(parents=True, exist_ok=True)

print("Opening remote GQA ZIP...")
print("This does NOT download the full 20 GB archive.")

with fsspec.open(
    URL,
    "rb",
    block_size=2 * 1024 * 1024,
    cache_type="readahead"
) as remote:

    with zipfile.ZipFile(remote) as z:

        matches = [
            n for n in z.namelist()
            if Path(n).stem == IMAGE_ID
        ]

        print("Matches:", matches)

        if not matches:
            raise RuntimeError("Image not found in official ZIP")

        member = matches[0]
        suffix = Path(member).suffix or ".jpg"
        dest = out / f"{IMAGE_ID}{suffix}"

        with z.open(member) as src:
            dest.write_bytes(src.read())

        print("Saved:", dest)
        print("Size:", round(dest.stat().st_size / 1024, 1), "KB")

print("DONE")
