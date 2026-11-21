import pandas as pd
from scipy.stats import ttest_rel
df = pd.read_csv(r"D:\experiment\results_cv.csv")
pairs = [("Gemma-2-2B QLoRA in-domain", "DeBERTaV3 in-domain"),
         ("Qwen2-1.5B QLoRA in-domain", "DeBERTaV3 in-domain"),
         ("Gemma-2-2B QLoRA in-domain", "Qwen2-1.5B QLoRA in-domain")]
for a, b in pairs:
    x = df[df["name"]==a].sort_values("fold")["f1"].values
    y = df[df["name"]==b].sort_values("fold")["f1"].values
    t, p = ttest_rel(x, y)
    print(f"{a}  vs  {b}\n   t={t:.3f}  p={p:.4f}\n")
