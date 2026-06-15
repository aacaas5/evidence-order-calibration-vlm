import requests
from pathlib import Path

IMAGE_ID = "2379780"

API = "https://datasets-server.huggingface.co/filter"

params = {
    "dataset": "alexwww94/GQA",
    "config": "val_balanced_images",
    "split": "train",
    "where": f'"id"=\'{IMAGE_ID}\'',
    "offset": 0,
    "length": 1,
}

print("Testing image:", IMAGE_ID)

r = requests.get(API, params=params, timeout=60)

print("HTTP:", r.status_code)

if r.status_code != 200:
    print(r.text[:1000])
    raise SystemExit

data = r.json()

print("Rows returned:", len(data.get("rows", [])))

if data.get("rows"):
    row = data["rows"][0]["row"]
    print("Keys:", list(row.keys()))
    print("ID:", row.get("id"))
    print("Image field:", row.get("image"))
