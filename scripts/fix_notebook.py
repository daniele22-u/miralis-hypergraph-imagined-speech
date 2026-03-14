import json
import sys

nb_path = "notebooks/EEG_04_braindecode_raw_baselines.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

fixed_count = 0
for i, cell in enumerate(nb["cells"]):
    if "source" in cell:
        src = cell["source"]
        if isinstance(src, list) and len(src) > 10 and all(len(s) <= 2 for s in src[:10]):
            res = "".join(s[0] for s in src)
            lines = [line + "\n" for line in res.split("\n")]
            if lines:
                if lines[-1] == "\n":
                    lines.pop()
                else:
                    lines[-1] = lines[-1][:-1]
            cell["source"] = lines
            fixed_count += 1
            print(f"Fixed cell {i}")

if fixed_count > 0:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved {nb_path} with {fixed_count} cells fixed.")
else:
    print("No corrupted cells found.")
