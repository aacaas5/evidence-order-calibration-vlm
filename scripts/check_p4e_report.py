import json
from pathlib import Path
from collections import Counter

p = Path("results/raw/p4e_selective_download_report.json")

data = json.loads(p.read_text(encoding="utf-8"))

print("Top-level type:", type(data).__name__)
print("Top-level keys:", list(data.keys()))

rows = data.get("results", [])

print("\nNumber of result rows:", len(rows))
print("Status counts:", Counter(r.get("status") for r in rows))

print("\nFirst 3 rows:")
for r in rows[:3]:
    print(r)
