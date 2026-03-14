import os
path = "notebooks/EEG_04_braindecode_raw_baselines.ipynb"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()
text = text.replace("CBraMod", "Labram")
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Notebook replaced CBraMod with Labram successfully")
