import json
import time
import base64
from pathlib import Path
from urllib.parse import urlencode

import requests


# ============================================================
# CONFIG
# ============================================================

MANIFEST = Path(
    "data/gqa/manifests/gqa_evidence_pilot_50.json"
)

OUT_DIR = Path(
    "data/gqa/pilot_images"
)

REPORT_PATH = Path(
    "results/raw/p4e_selective_download_report.json"
)

DATASET = "Mineru/GQA"
CONFIG = "default"
SPLIT = "val_balanced"

API = (
    "https://datasets-server.huggingface.co/filter"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK MANIFEST
# ============================================================

if not MANIFEST.exists():

    raise FileNotFoundError(
        f"\nMissing manifest:\n{MANIFEST}\n\n"
        "Run P4D first."
    )


with open(
    MANIFEST,
    "r",
    encoding="utf-8",
) as f:

    samples = json.load(f)


print("=" * 76)
print("PROJECT 3 - P4E LITE SELECTIVE GQA IMAGE FETCH")
print("=" * 76)

print(
    "\nPilot questions:",
    len(samples)
)

print(
    "Source:",
    DATASET,
    SPLIT,
)

print(
    "\nWe will NOT download the 20 GB GQA image archive."
)


# ============================================================
# IMAGE CELL DECODER
# ============================================================

def save_image_cell(image_cell, destination):
    """
    Hugging Face dataset-server image cells may be represented
    in slightly different ways.

    Handle:
      1. {"src": "https://..."}
      2. {"src": "data:image/jpeg;base64,..."}
      3. direct URL string
    """

    if isinstance(
        image_cell,
        dict,
    ):

        src = image_cell.get(
            "src"
        )

    else:

        src = image_cell


    if not src:

        raise RuntimeError(
            "No image source found in returned row."
        )


    # --------------------------------------------------------
    # Base64 data URL
    # --------------------------------------------------------

    if isinstance(
        src,
        str,
    ) and src.startswith(
        "data:image"
    ):

        header, encoded = src.split(
            ",",
            1,
        )

        image_bytes = base64.b64decode(
            encoded
        )

        destination.write_bytes(
            image_bytes
        )

        return


    # --------------------------------------------------------
    # Normal HTTP image URL
    # --------------------------------------------------------

    if isinstance(
        src,
        str,
    ) and src.startswith(
        ("http://", "https://")
    ):

        response = requests.get(
            src,
            timeout=60,
        )

        response.raise_for_status()

        destination.write_bytes(
            response.content
        )

        return


    raise RuntimeError(
        f"Unsupported image representation: {type(src)}"
    )


# ============================================================
# DOWNLOAD EACH QUESTION'S IMAGE
# ============================================================

results = []

downloaded_image_ids = set()


for i, sample in enumerate(
    samples,
    start=1,
):

    question_id = str(
        sample["question_id"]
    )

    image_id = str(
        sample["image_id"]
    )

    destination = (
        OUT_DIR /
        f"{image_id}.jpg"
    )


    print(
        f"\n[{i}/{len(samples)}] "
        f"QID={question_id} "
        f"IMG={image_id}"
    )


    # --------------------------------------------------------
    # Avoid re-downloading the same image if multiple questions
    # happen to use it.
    # --------------------------------------------------------

    if (
        destination.exists()
        and destination.stat().st_size > 1000
    ):

        print(
            "Already exists -> skipping"
        )

        downloaded_image_ids.add(
            image_id
        )

        results.append(
            {
                "question_id":
                    question_id,

                "image_id":
                    image_id,

                "status":
                    "already_exists",

                "bytes":
                    destination.stat().st_size,
            }
        )

        continue


    # --------------------------------------------------------
    # Filter using QUESTION ID.
    #
    # Mineru/GQA exposes question_id, question, answer, image.
    # --------------------------------------------------------

    where = (
        f"\"question_id\"='{question_id}'"
    )


    params = {
        "dataset":
            DATASET,

        "config":
            CONFIG,

        "split":
            SPLIT,

        "where":
            where,

        "offset":
            0,

        "length":
            1,
    }


    try:

        response = requests.get(
            API,
            params=params,
            timeout=90,
        )


        if response.status_code != 200:

            print(
                "Dataset server HTTP:",
                response.status_code
            )

            results.append(
                {
                    "question_id":
                        question_id,

                    "image_id":
                        image_id,

                    "status":
                        f"http_{response.status_code}",
                }
            )

            continue


        payload = response.json()

        rows = payload.get(
            "rows",
            []
        )


        if not rows:

            print(
                "No matching row found."
            )

            results.append(
                {
                    "question_id":
                        question_id,

                    "image_id":
                        image_id,

                    "status":
                        "not_found",
                }
            )

            continue


        row = rows[0].get(
            "row",
            {}
        )


        returned_question = str(
            row.get(
                "question",
                ""
            )
        )


        returned_answer = str(
            row.get(
                "answer",
                ""
            )
        )


        image_cell = row.get(
            "image"
        )


        print(
            "HF question:",
            returned_question[:80]
        )

        print(
            "HF answer:",
            returned_answer
        )


        save_image_cell(
            image_cell,
            destination,
        )


        size = destination.stat().st_size


        print(
            "Saved:",
            destination
        )

        print(
            "Size:",
            round(
                size / 1024,
                1,
            ),
            "KB"
        )


        downloaded_image_ids.add(
            image_id
        )


        results.append(
            {
                "question_id":
                    question_id,

                "image_id":
                    image_id,

                "status":
                    "downloaded",

                "bytes":
                    size,

                "hf_question":
                    returned_question,

                "hf_answer":
                    returned_answer,
            }
        )


    except Exception as exc:

        print(
            "ERROR:",
            type(exc).__name__,
            str(exc),
        )


        results.append(
            {
                "question_id":
                    question_id,

                "image_id":
                    image_id,

                "status":
                    "error",

                "error":
                    str(exc),
            }
        )


    # Gentle delay so we don't hammer the public endpoint.
    time.sleep(
        0.25
    )


# ============================================================
# SUMMARY
# ============================================================

successful = [
    r
    for r in results
    if r["status"]
    in {
        "downloaded",
        "already_exists",
    }
]


failed = [
    r
    for r in results
    if r["status"]
    not in {
        "downloaded",
        "already_exists",
    }
]


total_bytes = sum(
    r.get(
        "bytes",
        0,
    )
    for r in successful
)


report = {
    "dataset":
        DATASET,

    "split":
        SPLIT,

    "requested_questions":
        len(samples),

    "successful_rows":
        len(successful),

    "unique_images_available":
        len(downloaded_image_ids),

    "failed_rows":
        len(failed),

    "downloaded_megabytes":
        round(
            total_bytes / 1024**2,
            2,
        ),

    "results":
        results,
}


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 76)
print("P4E LITE COMPLETE")
print("=" * 76)

print(
    "Requested questions:",
    len(samples)
)

print(
    "Successful:",
    len(successful)
)

print(
    "Unique pilot images:",
    len(downloaded_image_ids)
)

print(
    "Failed:",
    len(failed)
)

print(
    "Total image data:",
    report[
        "downloaded_megabytes"
    ],
    "MB"
)

print(
    "\nImage folder:",
    OUT_DIR
)

print(
    "Report:",
    REPORT_PATH
)

if failed:

    print(
        "\nSome rows failed."
    )

    print(
        "That is okay - send me the failure statuses "
        "before we try another source."
    )
