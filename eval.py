import pandas as pd
from jiwer import cer

df = pd.read_csv("output_old.csv", encoding="utf-8-sig")

df["GroundTruth"] = df["GroundTruth"].fillna("").astype(str)
df["HanChar"] = df["HanChar"].fillna("").astype(str)

# Hàm kiểm tra xem chuỗi có chứa số không
def contains_number(s):
    return any(ch.isdigit() for ch in s)

# Loại bỏ những row có chứa số ở GT hoặc HanChar
df = df[~df["GroundTruth"].apply(contains_number)]
df = df[~df["HanChar"].apply(contains_number)]

# Hàm làm sạch chuỗi
def clean_text(s):
    s = s.replace('"', '')      # Xóa dấu "
    s = s.replace(" ", "")      # Xóa khoảng trắng
    return s

df["GroundTruth_clean"] = df["GroundTruth"].apply(clean_text)
df["HanChar_clean"] = df["HanChar"].apply(clean_text)

df["CER"] = df.apply(
    lambda row: cer(row["GroundTruth_clean"], row["HanChar_clean"]),
    axis=1
)

print("Remaining rows:", len(df))
print("Mean CER:", df["CER"].mean())
