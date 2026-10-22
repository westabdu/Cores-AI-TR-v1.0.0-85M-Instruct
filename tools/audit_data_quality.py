"""
Cores-AI TR — Veri Kümesi Kalite ve Temsili Örnek Denetimi
"""
import os
import sys
import json
import random

# Kök çalışma dizinine sabitle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

train_path = os.path.join("data", "sft_pilot", "train.jsonl")
assert os.path.exists(train_path), f"Train file not found: {train_path}"

samples = []
with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

print(f"Toplam Yüklenen Eğitim Örneği: {len(samples)}")

# Kaynaklara göre grupla
by_source = {}
for s in samples:
    src = s.get("source", "unknown")
    by_source.setdefault(src, []).append(s)

for src, slist in by_source.items():
    print(f"   Kaynak: {src} -> {len(slist)} örnek")

random.seed(42)

# Her kaynaktan 5'er adet olmak üzere toplam 15 temsilci örnek seç
selected_15 = []
for src in ["sixfingerdev", "ogulcanaydogan", "AlicanKiraz0"]:
    pool = by_source.get(src, [])
    if len(pool) >= 5:
        chosen = random.sample(pool, 5)
    else:
        chosen = pool
    selected_15.extend(chosen)

print(f"\nSeçilen Temsili Örnek Sayısı: {len(selected_15)}")

# Örnekleri listele
for idx, s in enumerate(selected_15, 1):
    src = s.get("source", "unknown")
    msgs = s.get("messages", [])
    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    print(f"\n{'='*70}")
    print(f"ÖRNEK {idx:02d} / 15 | Kaynak: {src}")
    print(f"{'='*70}")
    print(f"[KULLANICI]: {user_msg}")
    print(f"[ASİSTAN]  : {asst_msg}")
