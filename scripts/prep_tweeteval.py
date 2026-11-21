import os
from datasets import load_dataset

os.makedirs(r"D:\experiment\data", exist_ok=True)

ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
LBL = {0: "negative", 1: "neutral", 2: "positive"}

def to_binary(split, fname):
    df = ds[split].to_pandas()
    print(split, ":", df["label"].map(LBL).value_counts().to_dict())
    df = df[df["label"] != 1].copy()                    # drop neutral -> binary
    df["label"] = df["label"].map({0: "negative", 2: "positive"})
    df[["text", "label"]].to_csv(rf"D:\experiment\data\{fname}", index=False)
    print("  saved:", fname, len(df), "rows")

to_binary("train", "train.csv")
to_binary("validation", "val.csv")
to_binary("test", "test.csv")
print("DONE")
