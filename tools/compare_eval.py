"""
Cores-AI TR — v1 Baseline vs v2 SFT Karşılaştırma Analiz Betiği
"""
import os
import sys
import json
import re

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

with open('TESTLER/eval_v1_baseline.json', 'r', encoding='utf-8') as f:
    v1_data = json.load(f)
with open('TESTLER/eval_v2_sft.json', 'r', encoding='utf-8') as f:
    v2_data = json.load(f)
with open('TESTLER/benchmark_questions.json', 'r', encoding='utf-8') as f:
    bm_data = json.load(f)

v1_res = {r['id']: r for r in v1_data['results']}
v2_res = {r['id']: r for r in v2_data['results']}

wiki_patterns = [r'\|\-', r'colspan', r'style=', r'{\|\s*class', r'Kaynakça', r'Dış bağlantılar', r'Kategori:']

def check_wiki(text):
    for p in wiki_patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def check_repetition(text):
    words = text.split()
    if len(words) < 6:
        return False
    for n in [2, 3, 4]:
        for i in range(len(words) - 2 * n):
            chunk1 = ' '.join(words[i:i+n])
            chunk2 = ' '.join(words[i+n:i+2*n])
            chunk3 = ' '.join(words[i+2*n:i+3*n]) if i+3*n <= len(words) else ''
            if chunk1.lower() == chunk2.lower() == chunk3.lower() and len(chunk1) > 4:
                return True
    return False

cats = bm_data['categories']

print("=" * 70)
print("GENEL METRİKLER (150 SORULUK SABİT BENCHMARK)")
print("=" * 70)
print(f"Toplam Soru Sayısı      : 150")
print(f"v1 Baseline Checkpoint  : {v1_data['meta']['model_checkpoint']} (iter={v1_data['meta']['iter_num']})")
print(f"v2 SFT Checkpoint       : {v2_data['meta']['model_checkpoint']} (iter={v2_data['meta']['iter_num']})")
print(f"v1 Val Loss             : {v1_data['meta']['best_val_loss']:.4f}")
print(f"v2 Val Loss (SFT)       : {v2_data['meta']['val_loss']:.4f}")
print(f"v1 EOS Hit Rate         : {v1_data['meta']['eos_hit_count']}/150 ({v1_data['meta']['eos_hit_rate']})")
print(f"v2 EOS Hit Rate         : {v2_data['meta']['eos_hit_count']}/150 ({v2_data['meta']['eos_hit_rate']})")
print(f"v1 Ort. Süre / Soru     : {v1_data['meta']['elapsed_seconds']/150:.2f} saniye")
print(f"v2 Ort. Süre / Soru     : {v2_data['meta']['elapsed_seconds']/150:.2f} saniye")
print(f"v2 Ort. Üretilen Token  : {v2_data['meta']['avg_generated_tokens']}")

v1_wiki = sum(1 for r in v1_res.values() if check_wiki(r['output']))
v2_wiki = sum(1 for r in v2_res.values() if check_wiki(r['assistant_response']))
print(f"Vikipedi / Tablo Kodu   : v1={v1_wiki}/150 ({v1_wiki/1.5:.1f}%) -> v2={v2_wiki}/150 ({v2_wiki/1.5:.1f}%)")

v1_rep = sum(1 for r in v1_res.values() if check_repetition(r['output']))
v2_rep = sum(1 for r in v2_res.values() if check_repetition(r['assistant_response']))
print(f"Ağır Tekrar Döngüsü     : v1={v1_rep}/150 ({v1_rep/1.5:.1f}%) -> v2={v2_rep}/150 ({v2_rep/1.5:.1f}%)")

print("\n" + "=" * 70)
print("KATEGORİ BAZINDA EOS VURMA ORANLARI")
print("=" * 70)
print(f"{'Kat':<4} | {'Kategori Adı':<30} | {'Soru':<5} | {'v1 EOS %':<10} | {'v2 EOS %':<10} | {'Fark'}")
print("-" * 70)
for cid, cinfo in cats.items():
    v1_c_eos = sum(1 for r in v1_res.values() if r['category'] == cid and r['eos_found'])
    v2_c_eos = sum(1 for r in v2_res.values() if r['category'] == cid and r['eos_found'])
    tot = cinfo['count']
    p1 = v1_c_eos / tot * 100
    p2 = v2_c_eos / tot * 100
    diff = p2 - p1
    diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
    print(f"{cid:<4} | {cinfo['name']:<30} | {tot:<5} | %{p1:<9.1f} | %{p2:<9.1f} | {diff_str}")

print("\n" + "=" * 70)
print("KATEGORİLERDEN TEMSİLİ ÖRNEKLER (v1 vs v2)")
print("=" * 70)

sample_ids = [
    "A01", "A02", "A03", "A07", "A10",  # Kategori A
    "B01", "B05", "B10",                 # Kategori B
    "C01", "C02", "C05", "C08",          # Kategori C
    "D01", "D03", "D10",                 # Kategori D
    "E01", "E05",                        # Kategori E
    "F01", "F05"                         # Kategori F
]

for sid in sample_ids:
    matching_q = [x for x in bm_data['questions'] if x['id'] == sid][0]
    prompt = matching_q['prompt']
    cat = matching_q['category']
    v1_out = v1_res[sid]['output'].strip()
    v2_out = v2_res[sid]['assistant_response'].strip()
    v1_eos = "EOS: EVET" if v1_res[sid]['eos_found'] else "EOS: HAYIR (128 tok kesildi)"
    v2_eos = "EOS: EVET" if v2_res[sid]['eos_found'] else "EOS: HAYIR (128 tok kesildi)"
    toks2 = v2_res[sid]['generated_tokens']

    print(f"\n--- [{sid} - Kat: {cat}] Prompt: {prompt!r} ---")
    print(f"  [v1 Baseline] ({v1_eos}):")
    for line in v1_out.splitlines()[:4]:
        print(f"    {line}")
    if len(v1_out.splitlines()) > 4:
        print("    ...")
    print(f"  [v2 SFT Pilot] ({v2_eos}, {toks2} tok):")
    for line in v2_out.splitlines()[:4]:
        print(f"    {line}")
    if len(v2_out.splitlines()) > 4:
        print("    ...")
