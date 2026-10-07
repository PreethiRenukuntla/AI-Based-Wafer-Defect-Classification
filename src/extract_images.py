import os
import sys
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image

DATA_PATH = r"data\raw\wm811k\LSWMD.pkl"

OUTPUT_DIR = r"data\processed\images"

METADATA_PATH = (
    r"data\processed\selected_metadata.csv"
)

MAX_PER_CLASS = 5000

IMAGE_SIZE = 64


import pandas.core.indexes.base
import pandas.core.indexes.range
import pandas.core.indexes.category

sys.modules["pandas.indexes"] = pd.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base
sys.modules["pandas.indexes.range"] = pandas.core.indexes.range
sys.modules["pandas.indexes.category"] = pandas.core.indexes.category


def clean_label(value):

    if isinstance(value, list):

        if len(value) == 0:
            return "Unknown"

        value = value[0]

    if pd.isna(value):
        return "Unknown"

    value = str(value)

    value = value.replace("[", "")
    value = value.replace("]", "")
    value = value.replace("'", "")
    value = value.strip()

    if value.lower() == "none":
        return "Normal"

    return value


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Loading WM-811K...")

df = pd.read_pickle(DATA_PATH)

df["label"] = (
    df["failureType"]
    .apply(clean_label)
)


valid_labels = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Normal",
    "Random",
    "Scratch"
]


selected = defaultdict(int)

metadata_rows = []

successful = 0
failed = 0


for index, row in df.iterrows():

    label = row["label"]

    if label not in valid_labels:
        continue

    if selected[label] >= MAX_PER_CLASS:
        continue

    wafer_map = row["waferMap"]

    try:

        array = np.array(
            wafer_map,
            dtype=np.uint8
        )

        image = Image.fromarray(
            array
        ).convert("L")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        class_dir = os.path.join(
            OUTPUT_DIR,
            label
        )

        os.makedirs(
            class_dir,
            exist_ok=True
        )

        filename = (
            f"{label}_{index}.png"
        )

        filepath = os.path.join(
            class_dir,
            filename
        )

        image.save(filepath)

        selected[label] += 1

        metadata_rows.append({
            "original_index": index,
            "label": label,
            "filename": filename,
            "path": filepath,
            "lotName": row["lotName"],
            "waferIndex": row["waferIndex"]
        })

        successful += 1

    except Exception as error:

        failed += 1

        print(
            f"Failed index {index}: {error}"
        )


metadata = pd.DataFrame(
    metadata_rows
)

metadata.to_csv(
    METADATA_PATH,
    index=False
)


print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

for label in valid_labels:

    print(
        f"{label:12s}: {selected[label]}"
    )

print()
print("Successful:", successful)
print("Failed:", failed)

print()
print("Images saved to:")
print(OUTPUT_DIR)

print()
print("Metadata saved to:")
print(METADATA_PATH)