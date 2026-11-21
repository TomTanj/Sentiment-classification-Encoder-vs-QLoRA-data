import pandas as pd
from scipy.stats import ttest_rel

df = pd.read_csv(r"D:\experiment\results_cv.csv")
g = df.groupby("name").agg(
    acc_mean=("acc","mean"), acc_std=("acc","std"),
    f1_mean=("f1","mean"),  f1_std=("f1","std"),
    folds=("fold","count")).sort_values("f1_mean", ascending=False)
print(g.round(4).to_string())

top2 = g.head(2).index.tolist()
if len(top2) == 2:
    a = df[df["name"]==top2[0]].sort_values("fold")["f1"].values
    b = df[df["name"]==top2[1]].sort_values("fold")["f1"].values
    t, p = ttest_rel(a, b)
    print(f"\npaired t-test (top-2): t={t:.3f}  p={p:.4f}")
    print("p<0.05 = significant; with 5 folds treat as indicative, not proof")
