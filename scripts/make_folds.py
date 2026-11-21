import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv(r"D:\experiment\data-ai\test.csv")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for k, (tr, te) in enumerate(skf.split(df, df["label"])):
    df.iloc[tr][["text","label"]].to_csv(rf"D:\experiment\data-ai\fold{k}_train.csv", index=False)
    df.iloc[te][["text","label"]].to_csv(rf"D:\experiment\data-ai\fold{k}_test.csv",  index=False)
    print(f"fold {k}: train={len(tr)}  test={len(te)}")
print("folds saved - identical for every model = fair comparison")
