import requests, json
from pathlib import Path

manifest = json.loads(
    Path("data/gqa/manifests/gqa_evidence_pilot_50.json").read_text(encoding="utf-8")
)

ids = [str(x["image_id"]) for x in manifest]
test_ids = ids[:5]

datasets = [
    ("Voxel51/GQA-Scene-Graph", "default"),
    ("wliafe/GQA200", "default"),
]

API = "https://datasets-server.huggingface.co/filter"

for dataset, config in datasets:
    print("\nDATASET:", dataset)

    for image_id in test_ids:
        params = {
            "dataset": dataset,
            "config": config,
            "split": "train",
            "where": f'"id"=\'{image_id}\'',
            "offset": 0,
            "length": 1,
        }

        try:
            r = requests.get(API, params=params, timeout=30)
            print(image_id, "HTTP", r.status_code)

            if r.status_code == 200:
                rows = r.json().get("rows", [])
                print("  rows:", len(rows))
        except Exception as e:
            print(" ", type(e).__name__, e)
