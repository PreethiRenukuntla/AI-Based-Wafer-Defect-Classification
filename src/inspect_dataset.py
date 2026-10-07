import sys
import pandas as pd

DATA_PATH = r"data\raw\wm811k\LSWMD.pkl"

import pandas.core.indexes.base
import pandas.core.indexes.range
import pandas.core.indexes.category

sys.modules["pandas.indexes"] = pd.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base
sys.modules["pandas.indexes.range"] = pandas.core.indexes.range
sys.modules["pandas.indexes.category"] = pandas.core.indexes.category

df = pd.read_pickle(DATA_PATH)

print("=" * 60)
print("WM-811K DATASET INSPECTION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

print("\nFailure type examples:")
print(df["failureType"].head(20))

print("\nTraining/Test label examples:")
print(df["trianTestLabel"].head(20))