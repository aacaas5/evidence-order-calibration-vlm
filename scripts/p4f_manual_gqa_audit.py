import json, tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

MANIFEST = Path("data/gqa/manifests/gqa_evidence_pilot_50.json")
IMAGE_DIR = Path("data/gqa/pilot_images")
PROGRESS = Path("data/gqa/manifests/gqa_evidence_audit_progress.json")
AUDITED = Path("data/gqa/manifests/gqa_evidence_pilot_audited.json")

samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
decisions = {}
idx = 0
photo = None

if PROGRESS.exists():
    old = json.loads(PROGRESS.read_text(encoding="utf-8"))
    decisions = {x["question_id"]: x.get("audit_status", "pending") for x in old}

def find_image(image_id):
    matches = list(IMAGE_DIR.glob(f"{image_id}.*"))
    if not matches:
        return None
    return matches[0]

def save():
    rows, accepted = [], []

    for s in samples:
        x = dict(s)
        x["audit_status"] = decisions.get(s["question_id"], "pending")
        rows.append(x)

        if x["audit_status"] == "accept":
            accepted.append(x)

    PROGRESS.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    AUDITED.write_text(json.dumps(accepted, indent=2), encoding="utf-8")

def show():
    global photo

    s = samples[idx]
    obj = s["critical_objects"][0]

    q_label.config(text=f"{idx+1}/{len(samples)}   Q: {s['question']}")
    a_label.config(text=f"Answer: {s['answer']}")
    o_label.config(text=f"Critical object: {obj['name']}")

    path = find_image(str(s["image_id"]))

    if path is None:
        image_label.config(image="", text=f"IMAGE NOT FOUND: {s['image_id']}")
        status_label.config(text="Current: MISSING IMAGE")
        return

    img = Image.open(path).convert("RGB")

    x1, y1, x2, y2 = obj["bbox_xyxy"]
    draw = ImageDraw.Draw(img)

    for off in range(4):
        draw.rectangle(
            [x1-off, y1-off, x2+off, y2+off],
            outline="red"
        )

    img.thumbnail((900, 560))
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo, text="")

    status = decisions.get(s["question_id"], "pending")
    status_label.config(text=f"Current: {status.upper()}")

def decide(status):
    global idx

    decisions[samples[idx]["question_id"]] = status
    save()

    if idx < len(samples) - 1:
        idx += 1

    show()

def move(step):
    global idx
    idx = max(0, min(len(samples)-1, idx + step))
    show()

root = tk.Tk()
root.title("GQA Evidence Audit")
root.geometry("1050x800")

q_label = tk.Label(root, font=("Arial", 15, "bold"), wraplength=950)
q_label.pack(pady=8)

a_label = tk.Label(root, font=("Arial", 13))
a_label.pack()

o_label = tk.Label(root, font=("Arial", 13))
o_label.pack()

status_label = tk.Label(root, font=("Arial", 12))
status_label.pack(pady=4)

image_label = tk.Label(root)
image_label.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Button(frame, text="ACCEPT", width=15, command=lambda: decide("accept")).grid(row=0,column=0,padx=5)
tk.Button(frame, text="REJECT", width=15, command=lambda: decide("reject")).grid(row=0,column=1,padx=5)
tk.Button(frame, text="UNSURE", width=15, command=lambda: decide("unsure")).grid(row=0,column=2,padx=5)

root.bind("a", lambda e: decide("accept"))
root.bind("r", lambda e: decide("reject"))
root.bind("u", lambda e: decide("unsure"))
root.bind("<Left>", lambda e: move(-1))
root.bind("<Right>", lambda e: move(1))

show()
root.mainloop()

save()

counts = {"accept":0, "reject":0, "unsure":0, "pending":0}

for s in samples:
    counts[decisions.get(s["question_id"], "pending")] += 1

print("\nAUDIT SUMMARY")
for k, v in counts.items():
    print(f"{k}: {v}")
