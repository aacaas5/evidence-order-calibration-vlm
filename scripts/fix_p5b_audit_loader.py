from pathlib import Path

p = Path("scripts/p5b_run_gqa_pilot.py")
text = p.read_text(encoding="utf-8")

old = '''        decision = str(
            sample.get(
                "audit",
                sample.get(
                    "decision",
                    sample.get("audit_decision", "")
                )
            )
        ).lower()
'''

new = '''        decision = str(
            sample.get(
                "audit_status",
                sample.get(
                    "audit",
                    sample.get(
                        "decision",
                        sample.get("audit_decision", "")
                    )
                )
            )
        ).lower()
'''

if old not in text:
    raise RuntimeError("Expected loader block not found.")

p.write_text(text.replace(old, new), encoding="utf-8")

print("Patched P5B loader successfully.")
