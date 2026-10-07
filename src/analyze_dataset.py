import sys
from collections import Counter

import pandas as pd

DATA_PATH = r"data\raw\wm811k\LSWMD.pkl"
OUTPUT_PATH = r"data\processed\wm811k_metadata.csv"

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

    if value.lower() in ["none", ""]:
        return "none"

    return value


print("Loading WM-811K...")

df = pd.read_pickle(DATA_PATH)

df["failureType_clean"] = (
    df["failureType"].apply(clean_label)
)

df["split_clean"] = (
    df["trianTestLabel"].apply(clean_label)
)


print("\nDataset shape:")
print(df.shape)


print("\nFailure type distribution:")

counts = Counter(
    df["failureType_clean"]
)

for label, count in counts.most_common():

    print(
        f"{label:12s} {count}"
    )


print("\nTrain/Test distribution:")

split_counts = Counter(
    df["split_clean"]
)

for label, count in split_counts.items():

    print(
        f"{label:12s} {count}"
    )


metadata = df[
    [
        "lotName",
        "waferIndex",
        "dieSize",
        "failureType_clean",
        "split_clean"
    ]
]

metadata.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)