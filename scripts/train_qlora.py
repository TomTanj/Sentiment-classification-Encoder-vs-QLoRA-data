import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==================== CONFIG ====================
MODEL = "unsloth/gemma-2-2b-it-bnb-4bit"
OUT   = r"D:\experiment\models\gemma-2-2b-qlora"
TRAIN_LIMIT = 0    # FULL training, all 24,942 rows
BATCH  = 4
ACCUM  = 8
LR     = 2e-4
EPOCHS = 1
MAXLEN = 512
SEED   = 42
# ================================================

LABELS = ["negative", "positive"]

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=q, device_map={"": 0},
    torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False   # add this line too (QLoRA standard; prevents a warning/error with gradient checkpointing)

# LoRA: same prompt format as zeroshot_llm.py (must match!)
PROMPT = ("Classify the sentiment of the following text as positive or negative. "
          "Reply with exactly one word: positive or negative.\n\nText: {t}")

tid = {}
for lab in LABELS:
    ids = tok(lab, add_special_tokens=False).input_ids
    assert len(ids) == 1, f"{lab} is multi-token"
    tid[lab] = ids[0]
print("label tokens:", tid)

class ClsDataset(Dataset):
    """input = chat prompt + answer token; labels mask everything except the answer."""
    def __init__(self, df):
        self.items = []
        for t, l in zip(df["text"].astype(str), df["label"]):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(t=t[:2000])}],
                tokenize=True, add_generation_prompt=True)[:MAXLEN]
            ans = [tid[l], tok.eos_token_id]
            self.items.append({"input_ids": ids + ans,
                               "labels": [-100]*len(ids) + ans})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return {"input_ids": torch.tensor([b["input_ids"] + [pad]*(n-len(b["input_ids"])) for b in batch]),
            "labels":   torch.tensor([b["labels"] + [-100]*(n-len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1]*len(b["input_ids"]) + [0]*(n-len(b["input_ids"])) for b in batch])}

lcfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()   # expect ~0.5-1% trainable

train_df = pd.read_csv(r"D:\experiment\data\train.csv")
if TRAIN_LIMIT: train_df = train_df.sample(TRAIN_LIMIT, random_state=SEED)
print("training on", len(train_df), "rows")

args = TrainingArguments(
    output_dir=OUT, num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
    warmup_ratio=0.05, weight_decay=0.01, lr_scheduler_type="cosine",
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit", fp16=True, logging_steps=20,
    save_strategy="no", report_to="none", seed=SEED)

trainer = Trainer(model=model, args=args,
                  train_dataset=ClsDataset(train_df), data_collator=collate)
trainer.train()

model.save_pretrained(OUT)   # saves ONLY the small LoRA adapter
tok.save_pretrained(OUT)
print("adapter saved to", OUT)

# ---------- evaluate: same logprob scoring as zero-shot ----------
model.eval()
tok.padding_side = "left"
B = 16

@torch.no_grad()
def classify(texts):
    preds = []
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(t=t)}],
            tokenize=False, add_generation_prompt=True) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        logp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
        scores = torch.stack([logp[:, tid[l]] for l in LABELS], dim=1)
        preds += [LABELS[j] for j in scores.argmax(1).tolist()]
    return preds

for split in ["val", "test"]:
    df = pd.read_csv(rf"D:\experiment\data\{split}.csv")
    t0 = time.time()
    p = classify(df["text"].astype(str).tolist())
    print(f"\n=== gemma-2-2b-QLoRA | {split} | {len(df)} rows ===")
    print("accuracy :", round(accuracy_score(df["label"], p), 4))
    print("f1_macro :", round(f1_score(df["label"], p, average="macro"), 4))
    print("confusion:", LABELS)
    print(confusion_matrix(df["label"], p, labels=LABELS))
    print("runtime  :", round(time.time()-t0), "s")
