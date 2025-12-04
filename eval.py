import pandas as pd
from jiwer import cer

df = pd.read_csv("output.csv", encoding="utf-8-sig")

df["GroundTruth"] = df["GroundTruth"].fillna("").astype(str)
df["HanChar"] = df["HanChar"].fillna("").astype(str)

df["CER"] = df.apply(
    lambda row: cer(row["GroundTruth"], row["HanChar"]),
    axis=1
)

print("Mean CER:", df["CER"].mean())
