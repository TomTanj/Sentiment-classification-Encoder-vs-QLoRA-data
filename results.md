# Results of Sentiment Classification: Encoders vs Open-Weight LLMs

* Hardware: RTX 2060 Mobile 6GB | torch 2.5.1+cu121 | transformers 4.48.3 | seed 42
* Label protocol: LLMs scored by label-token log-probability (deterministic); encoders via classification head
* Mistral-7B/Mixtral/Gemma-9B excluded: VRAM limitation (documented)

## Chapter 1 - Benchmark: TweetEval binary (train 24,942 / test 6,347)

|Model|Method|Acc|F1 macro|
|-|-|-|-|
|Qwen2-1.5B|zero-shot|84.04|80.94|
|ELECTRA-base|fine-tune|92.88|92.51|
|Gemma-2-2B|zero-shot|93.40|92.82|
|DeBERTaV3-base|fine-tune|94.86|94.54|
|Gemma-2-2B|QLoRA 4k rows (0.79% params)|95.18|94.85|
|Gemma-2-2B|QLoRA full data|**95.51**|**95.25**|

## Chapter 2 - Domain transfer to AI-domain data (n=1,043, single split)

|Model|Method|Acc|F1 macro|
|-|-|-|-|
|ELECTRA|transfer (TweetEval-trained)|75.26|72.22|
|Qwen2-1.5B|zero-shot|77.76|70.05|
|DeBERTaV3|transfer|79.39|75.87|
|Gemma-2-2B|zero-shot|**82.45**|**79.17**|
|Gemma-2-2B|QLoRA (TweetEval-trained)|80.63|78.25|

Key: zero-shot Gemma beat ALL out-of-domain fine-tuned models; TweetEval-trained
QLoRA scored BELOW its own zero-shot base (negative transfer).

## Chapter 3 - In-domain comparison: 5-fold stratified CV (n=1,043)

Identical folds for all models; in-domain methods trained on \~834 rows/fold.

|Model|Method|Acc (mean±std)|F1 (mean±std)|
|-|-|-|-|
|Gemma-2-2B QLoRA in-domain|in-domain QLoRA (\~834 rows, 0.79% params)|88.78 ± 2.11|87.51 ± 2.35|
|Qwen2-1.5B QLoRA in-domain|in-domain QLoRA (\~834 rows, 1.18% params)|86.96 ± 1.59|85.27 ± 1.83|
|DeBERTaV3 in-domain|in-domain full fine-tune (\~834 rows)|83.70 ± 3.31|82.22 ± 3.47|
|Gemma-2-2B zero-shot|zero-shot (no training)|82.55 ± 3.17|79.25 ± 3.85|
|ELECTRA in-domain|in-domain full fine-tune (\~834 rows)|81.49 ± 3.06|79.36 ± 3.32|
|DeBERTaV3 transfer|fine-tuned on TweetEval|79.38 ± 2.74|75.88 ± 3.05|
|Qwen2-1.5B zero-shot|zero-shot (no training)|77.76 ± 3.51|69.86 ± 5.74|
|ELECTRA transfer|fine-tuned on TweetEval|75.26 ± 3.15|72.25 ± 3.14|

|Statistical tests (paired t-test on macro-F1, 5 folds)|
|-|

|Gemma QLoRA vs DeBERTaV3 in-domain|t=4.707, p=0.0093 (significant)|
|-|-|
|Qwen QLoRA vs DeBERTaV3 in-domain|t=3.473, p=0.0255 (significant)|
|Gemma QLoRA vs Qwen QLoRA|t=2.050, p=0.1097 (n.s. — statistically tied)|



## Findings

1. Group 2 + PEFT champion: Gemma-2-2B QLoRA 88.78, besting best encoder by \~5 pts (5/5 folds)
2. Both LLM QLoRA models (86.96, 88.78) beat the best encoder (83.70)
3. Gemma zero-shot (82.55) \~ DeBERTa with 800 labels (83.70): LLM prior ≈ 800 labeled examples
4. LoRA repaired Qwen zero-shot class bias: acc-F1 gap 7.90 -> 1.69
5. DeBERTa > ELECTRA in every setting; ELECTRA fastest per fold (27s vs 44s)
6. Decision matrix: cold start -> zero-shot Gemma; \~800 labels -> in-domain QLoRA;
wrong-domain fine-tuning worse than doing nothing



=========================================================================================

First Data

&#x20;                     acc\_mean  acc\_std  f1\_mean  f1\_std  folds

name

Gemma-2-2B zero-shot    0.8255   0.0317   0.7925  0.0385      5

DeBERTaV3 transfer      0.7938   0.0274   0.7588  0.0305      5

ELECTRA transfer        0.7526   0.0315   0.7225  0.0314      5

Qwen2-1.5B zero-shot    0.7776   0.0351   0.6986  0.0574      5



=========================================================================================

Second Data

&#x20;                           acc\_mean  acc\_std  f1\_mean  f1\_std  folds

name

Gemma-2-2B QLoRA in-domain    0.8878   0.0211   0.8751  0.0235      5

Qwen2-1.5B QLoRA in-domain    0.8696   0.0159   0.8527  0.0183      5

DeBERTaV3 in-domain           0.8370   0.0331   0.8222  0.0347      5

ELECTRA in-domain             0.8149   0.0306   0.7936  0.0332      5

Gemma-2-2B zero-shot          0.8255   0.0317   0.7925  0.0385      5

DeBERTaV3 transfer            0.7938   0.0274   0.7588  0.0305      5

ELECTRA transfer              0.7526   0.0315   0.7225  0.0314      5

Qwen2-1.5B zero-shot          0.7776   0.0351   0.6986  0.0574      5

