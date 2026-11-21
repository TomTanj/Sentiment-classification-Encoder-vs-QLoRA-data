import pandas as pd

# Try the full-metadata workbook first; fall back to the 3-column csv
CANDIDATES = [r"D:\experiment\mydata\sentiment analysis processed data.xlsx",
              r"D:\experiment\mydata\Sentiment Analysis Processed Data.xlsx"]
CANDIDATES += [f for f in __import__("glob").glob(r"D:\experiment\mydata\*.xlsx")]

raw = None
for path in CANDIDATES:
    try:
        raw = pd.ExcelFile(path, engine="openpyxl").parse(0)
        if raw.shape[1] >= 9:
            print("using:", path, raw.shape)
            break
        raw = None
    except Exception:
        continue

if raw is None:
    raise SystemExit("No metadata workbook found in D:\\experiment\\mydata - "
                     "copy the 'Sentiment Analysis Processed Data.xlsx' there and rerun")

raw.columns = [str(c).lower().replace("column1.", "").replace(" ", "_") for c in raw.columns]
print("columns:", list(raw.columns))

# map expected names defensively
colmap = {}
for c in raw.columns:
    if "tweet_text" in c: colmap[c] = "text"
    elif "positive" in c: colmap[c] = "label"
    elif "long" in c:     colmap[c] = "length"
    elif "likes" in c:    colmap[c] = "likes"
    elif "reposts" in c:  colmap[c] = "reposts"
    elif "comments" in c: colmap[c] = "comments"
    elif "timestamp" in c:colmap[c] = "ts"
raw = raw.rename(columns=colmap)

raw["label"]  = raw["label"].map({1: "positive", 2: "negative"})
raw["length"] = raw["length"].map({1: "long", 2: "short"})

print("\nn =", len(raw))
print("Class balance:", raw["label"].value_counts(normalize=True).round(3).to_dict())
print("Length:", raw["length"].value_counts(normalize=True).round(3).to_dict())

for col in ["likes", "reposts", "comments"]:
    if col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

print("\nEngagement by sentiment (medians):")
print(raw.groupby("label")[["likes","reposts","comments"]].median().to_string())
print("\nEngagement by length (medians):")
print(raw.groupby("length")[["likes","reposts"]].median().to_string())

if "ts" in raw.columns:
    print("\nDate range:", raw["ts"].min(), "->", raw["ts"].max())

print("\nTop 5 most-liked posts + labels:")
for _, r in raw.nlargest(5, "likes").iterrows():
    print(f"  [{r['label']:8s}] likes={int(r['likes']):6d}  {str(r['text'])[:80]}")

# bonus: engagement distribution is skewed - report medians AND means
print("\nLikes mean/median by sentiment:")
print(raw.groupby("label")["likes"].agg(["mean","median"]).round(1).to_string())
