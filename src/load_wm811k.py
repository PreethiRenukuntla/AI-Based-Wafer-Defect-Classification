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

print("Starting WM-811K load...")
print("Pandas:", pd.__version__)

df = pd.read_pickle(DATA_PATH)

print("\nSUCCESS")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3))