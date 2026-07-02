import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_250.json")
AUDITED = Path("data/gqa/manifests/gqa_evidence_scaled_250_codex_audited.json")
ACCEPTED = Path("data/gqa/manifests/gqa_evidence_scaled_accepted.json")
OUT_DIR = Path("results/p9c")
OUT_DIR.mkdir(parents=True, exist_ok=True)

reject = {
    1, 12, 24, 29, 33, 36, 37, 41, 42, 43, 44, 46, 49, 53, 56, 57,
    61, 62, 65, 68, 71, 72, 74, 75, 77, 81, 83, 84, 87, 89, 97, 99,
    105, 110, 111, 117, 119, 120, 123, 128, 129, 134, 135, 138, 144,
    148, 150, 171, 176, 187, 191, 194, 195, 198, 204, 205, 206, 209,
    216, 217, 219, 222, 224, 227, 228, 229, 234, 236, 240, 242, 248,
}
unsure = {26, 40, 147}

misplaced = {
    37, 42, 46, 89, 97, 105, 129, 134, 144, 205, 217, 219, 222, 224,
    227, 240,
}
weak = {
    33, 43, 44, 57, 61, 62, 65, 68, 71, 74, 75, 81, 84, 87, 119,
    148, 150, 171, 176, 187, 191, 194, 198, 206, 209, 229, 242,
}
ambiguous = {29, 111, 128}


def reason(index, status):
    if status == "accept":
        return (
            "Box clearly localizes the visible question-critical object; masking it "
            "meaningfully removes the main evidence."
        )
    if status == "unsure":
        return {
            26: "The small garbage-can region is visible, but its shape and localization are weak at this scale.",
            40: "The sweater region is a small sleeve fragment, so evidence removal may be incomplete.",
            147: "The door is visible only as a narrow edge; material evidence localization is debatable.",
        }[index]
    if index in misplaced:
        return "Box is misplaced or captures content that does not correspond to the named critical object."
    if index in weak:
        return "Critical object is too small, distant, occluded, or weakly visible for a reliable localized intervention."
    if index in ambiguous:
        return "Question/object reference is ambiguous or depends on broader/multiple visual evidence."
    return (
        "Box is too broad or fragmentary, or substantial equivalent evidence remains visible outside it."
    )


samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
if len(samples) != 250:
    raise RuntimeError(f"Expected 250 candidates, found {len(samples)}")

audited = []
for index, source in enumerate(samples):
    sample = dict(source)
    status = "reject" if index in reject else "unsure" if index in unsure else "accept"
    sample["audit_status"] = status
    sample["audit_reason"] = reason(index, status)
    sample["auditor"] = "codex_visual_audit"
    audited.append(sample)

AUDITED.write_text(json.dumps(audited, indent=2), encoding="utf-8")
accepted = [sample for sample in audited if sample["audit_status"] == "accept"]
ACCEPTED.write_text(json.dumps(accepted, indent=2), encoding="utf-8")

counts = Counter(sample["audit_status"] for sample in audited)
by_category = defaultdict(Counter)
for sample in audited:
    by_category[sample["category"]][sample["audit_status"]] += 1

summary = {
    "audit_type": "automated_visual_audit",
    "auditor": "codex_visual_audit",
    "total": len(audited),
    "accept": counts["accept"],
    "reject": counts["reject"],
    "unsure": counts["unsure"],
    "acceptance_rate": counts["accept"] / len(audited),
    "by_category": {
        category: {
            "total": sum(category_counts.values()),
            "accept": category_counts["accept"],
            "reject": category_counts["reject"],
            "unsure": category_counts["unsure"],
            "acceptance_rate": category_counts["accept"] / sum(category_counts.values()),
        }
        for category, category_counts in sorted(by_category.items())
    },
}
(OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

with (OUT_DIR / "audit_by_category.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["category", "total", "accept", "reject", "unsure", "acceptance_rate"],
    )
    writer.writeheader()
    for category, values in summary["by_category"].items():
        writer.writerow({"category": category, **values})

fields = [
    "question_id", "image_id", "category", "question", "answer",
    "audit_status", "audit_reason", "auditor",
]
for status, filename in (("reject", "rejected_examples.csv"), ("unsure", "unsure_examples.csv")):
    with (OUT_DIR / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample for sample in audited if sample["audit_status"] == status)

print("P9C AUTOMATED VISUAL AUDIT")
print("Total:", summary["total"])
print("Accept:", summary["accept"])
print("Reject:", summary["reject"])
print("Unsure:", summary["unsure"])
print("Acceptance rate:", f'{summary["acceptance_rate"]:.3f}')
for category, values in summary["by_category"].items():
    print(category, values)
print("Audited manifest:", AUDITED)
print("Accepted manifest:", ACCEPTED)
print("Results:", OUT_DIR)
