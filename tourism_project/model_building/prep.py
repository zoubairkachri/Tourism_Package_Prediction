import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")   # registered tourism.csv inside the data folder

# Drop the unnamed index column if the CSV export included one
df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])

# Drop the customer identifier column, it is not a predictive feature
df.drop(columns=["CustomerID"], inplace=True)

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values, so training and serving use the same representation.

target = "ProdTaken"                 # column to predict (1 if the customer purchased the package)
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
