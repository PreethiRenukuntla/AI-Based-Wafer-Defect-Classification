import sys
import pandas as pd
import numpy as np

DATA_PATH = r"data\raw\wm811k\LSWMD.pkl"

import pandas.core.indexes.base
import pandas.core.indexes.range
import pandas.core.indexes.category

sys.modules["pandas.indexes"] = pd.core.indexes
sys.modules["pandas.indexes.base"] = pandas.core.indexes.base
sys.modules["pandas.indexes.range"] = pandas.core.indexes.range
sys.modules["pandas.indexes.category"] = pandas.core.indexes.category

print("Loading dataset...")
df = pd.read_pickle(DATA_PATH)
print("Dataset loaded. Shape:", df.shape)

# Helper function to clean cell values from list-of-lists or ndarray
def clean_val(val):
    if isinstance(val, (list, np.ndarray)):
        if len(val) > 0:
            if isinstance(val[0], (list, np.ndarray)):
                if len(val[0]) > 0:
                    return str(val[0][0])
            return str(val[0])
        return ""
    return str(val)

print("\nExtracting failure types and trianTestLabel...")
failure_types = [clean_val(v) for v in df['failureType']]
trian_test_labels = [clean_val(v) for v in df['trianTestLabel']]

df_clean = pd.DataFrame({
    'trianTestLabel': trian_test_labels,
    'failureType': failure_types
})

print("\ntrianTestLabel counts:")
print(df_clean['trianTestLabel'].value_counts(dropna=False))

print("\nfailureType counts:")
print(df_clean['failureType'].value_counts(dropna=False))

print("\nCross-tabulation (failureType vs trianTestLabel):")
print(pd.crosstab(df_clean['failureType'], df_clean['trianTestLabel']))
