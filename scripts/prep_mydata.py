import pandas as pd

FILE = r"D:\experiment\mydata\yourfile.csv"      # <-- EDIT to your real path
OUT  = r"D:\experiment\data\test.csv"

df = pd.read_csv(FILE, encoding="cp1252")

# positional access = immune to column-name spelling
out = pd.DataFrame({
    "text":   df.iloc[:, 0].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
    "label":  df.iloc[:, 1].astype(int).map({1: "positive", 2: "negative"}),
    "length": df.iloc[:, 2].astype(int).map({1: "long", 2: "short"}),
})

before = len(out)
out = out.dropna(subset=["label"])
out = out[out["text"].str.len() > 0]
out = out.drop_duplicates(subset="text")
print(f"rows: {before} -> {len(out)} (dedupe + cleanup)")
print("label :", out["label"].value_counts().to_dict())
print("length:", out["length"].value_counts().to_dict())
for _, r in out.head(3).iterrows():
    print(f"  [{r.label}/{r.length}]", r.text[:90])

out.to_csv(OUT, index=False, encoding="utf-8")
print("saved:", OUT)
