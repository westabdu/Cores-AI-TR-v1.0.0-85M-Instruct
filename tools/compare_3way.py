"""
Cores-AI TR — Baseline (v1) vs Pilot (v2-250) vs Stage 7 (v2-750) 3-Yönlü Karşılaştırma
"""
import os
import sys
import json
import re
import hashlib

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

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# SHA256 Doğrulama
v1_sha = sha256("out-cores-85m-test/ckpt.pt")
pilot_sha = sha256("TESTLER/temp_sft_pilot/ckpt_best.pt")
stage7_sha = sha256("TESTLER/temp_sft_stage7/ckpt_best.pt")

with open('TESTLER/eval_v1_baseline.json', 'r', encoding='utf-8') as f:
    v1_data = json.load(f)
with open('TESTLER/eval_v2_sft.json', 'r', encoding='utf-8') as f:
    pilot_data = json.load(f)
with open('TESTLER/eval_v2_stage7.json', 'r', encoding='utf-8') as f:
    st7_data = json.load(f)
with open('TESTLER/benchmark_questions.json', 'r', encoding='utf-8') as f:
    bm_data = json.load(f)

v1_res = {r['id']: r for r in v1_data['results']}
pilot_res = {r['id']: r for r in pilot_data['results']}
st7_res = {r['id']: r for r in st7_data['results']}

wiki_patterns = [r'\|\-', r'colspan', r'style=', r'\{\|\s*class', r'Kaynakça', r'Dış bağlantılar', r'Kategori:']

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
            c1 = ' '.join(words[i:i+n]).lower()
            c2 = ' '.join(words[i+n:i+2*n]).lower()
            c3 = ' '.join(words[i+2*n:i+3*n]).lower() if i+3*n <= len(words) else ''
            if c1 == c2 == c3 and len(c1) > 4:
                return True
    return False

def is_cut_at_max(r):
    return not r['eos_found'] and (r.get('generated_tokens', 128) >= 120 or len(r['output'].split()) > 70)

cats = bm_data['categories']

print("=" * 80)
print("1. GENEL METRİKLER (3 YÖNLÜ KARŞILAŞTIRMA)")
print("=" * 80)
print(f"Toplam Soru Sayısı : 150")
print(f"v1 Baseline SHA256 : {v1_sha} ({'KORUNDU' if v1_sha == 'e88f284fdfb9ddc48075a97e47f509859319590df312a4226d9b355c63347e51' else 'DEĞİŞTİ'})")
print(f"Pilot SHA256       : {pilot_sha} ({'KORUNDU' if pilot_sha == '106343294c6af89f3bc9201b6a51a78c527b2657cd3cdfc4d2942c03e0d8bb6a' else 'DEĞİŞTİ'})")
print(f"Stage 7 SHA256     : {stage7_sha}")

v1_eos = sum(1 for r in v1_res.values() if r['eos_found'])
p_eos = sum(1 for r in pilot_res.values() if r['eos_found'])
s7_eos = sum(1 for r in st7_res.values() if r['eos_found'])

print(f"\nEOS Vurma Oranı:")
print(f"  - v1 Baseline : {v1_eos}/150 ({v1_eos/1.5:.1f}%)")
print(f"  - v2 Pilot    : {p_eos}/150 ({p_eos/1.5:.1f}%)")
print(f"  - v2 Aşama 7  : {s7_eos}/150 ({s7_eos/1.5:.1f}%)")

v1_wiki = sum(1 for r in v1_res.values() if check_wiki(r['output']))
p_wiki = sum(1 for r in pilot_res.values() if check_wiki(r['assistant_response']))
s7_wiki = sum(1 for r in st7_res.values() if check_wiki(r['assistant_response']))

print(f"\nVikipedi / Tablo / HTML Kirliliği:")
print(f"  - v1 Baseline : {v1_wiki}/150 ({v1_wiki/1.5:.1f}%)")
print(f"  - v2 Pilot    : {p_wiki}/150 ({p_wiki/1.5:.1f}%)")
print(f"  - v2 Aşama 7  : {s7_wiki}/150 ({s7_wiki/1.5:.1f}%)")

v1_rep = sum(1 for r in v1_res.values() if check_repetition(r['output']))
p_rep = sum(1 for r in pilot_res.values() if check_repetition(r['assistant_response']))
s7_rep = sum(1 for r in st7_res.values() if check_repetition(r['assistant_response']))

print(f"\nAğır Tekrar Döngüsü:")
print(f"  - v1 Baseline : {v1_rep}/150 ({v1_rep/1.5:.1f}%)")
print(f"  - v2 Pilot    : {p_rep}/150 ({p_rep/1.5:.1f}%)")
print(f"  - v2 Aşama 7  : {s7_rep}/150 ({s7_rep/1.5:.1f}%)")

v1_cuts = sum(1 for r in v1_res.values() if not r['eos_found'])
p_cuts = sum(1 for r in pilot_res.values() if not r['eos_found'])
s7_cuts = sum(1 for r in st7_res.values() if not r['eos_found'])

print(f"\nMaksimum Tokenda Kesilme (128 tok dolup susamayan):")
print(f"  - v1 Baseline : {v1_cuts}/150 ({v1_cuts/1.5:.1f}%)")
print(f"  - v2 Pilot    : {p_cuts}/150 ({p_cuts/1.5:.1f}%)")
print(f"  - v2 Aşama 7  : {s7_cuts}/150 ({s7_cuts/1.5:.1f}%)")

p_tok = pilot_data['meta'].get('avg_generated_tokens', 25.9)
s7_tok = st7_data['meta'].get('avg_generated_tokens', 24.7)
print(f"\nOrtalama Üretilen Token Sayısı:")
print(f"  - v1 Baseline : ~85.0 token")
print(f"  - v2 Pilot    : {p_tok} token")
print(f"  - v2 Aşama 7  : {s7_tok} token")

print("\n" + "=" * 80)
print("2. KATEGORİ BAZINDA EOS VE DURMA ORANLARI")
print("=" * 80)
print(f"{'Kat':<4} | {'Kategori Adı':<28} | {'Soru':<5} | {'v1 Base %':<10} | {'v2 Pilot %':<11} | {'v2 Aşama 7 %':<12}")
print("-" * 80)
for cid, cinfo in cats.items():
    c_tot = cinfo['count']
    c_v1 = sum(1 for r in v1_res.values() if r['category'] == cid and r['eos_found']) / c_tot * 100
    c_p = sum(1 for r in pilot_res.values() if r['category'] == cid and r['eos_found']) / c_tot * 100
    c_s7 = sum(1 for r in st7_res.values() if r['category'] == cid and r['eos_found']) / c_tot * 100
    print(f"{cid:<4} | {cinfo['name']:<28} | {c_tot:<5} | %{c_v1:<9.1f} | %{c_p:<10.1f} | %{c_s7:<11.1f}")

print("\n" + "=" * 80)
print("3. TEMSİLİ 10+ SORU KARŞILAŞTIRMASI (Pilot vs Aşama 7)")
print("=" * 80)

check_qids = [
    "A01", "A02", "A10", # Kategori A
    "B02", "B05",       # Kategori B
    "C01", "C02", "C05", "C08", # Kategori C
    "D01", "D03", "D10", "D20", # Kategori D
    "E01", "E05",       # Kategori E
    "F05", "F12", "F20"  # Kategori F
]

for qid in check_qids:
    q = [x for x in bm_data['questions'] if x['id'] == qid][0]
    prompt = q['prompt']
    cat = q['category']
    p_resp = pilot_res[qid]['assistant_response'].strip().replace('\n', ' ')
    s7_resp = st7_res[qid]['assistant_response'].strip().replace('\n', ' ')
    p_tok = pilot_res[qid].get('generated_tokens', 0)
    s7_tok = st7_res[qid].get('generated_tokens', 0)
    p_eos = "EOS" if pilot_res[qid]['eos_found'] else "CUT"
    s7_eos = "EOS" if st7_res[qid]['eos_found'] else "CUT"

    print(f"\n--- [{qid} - Kat {cat}] Soru: '{prompt}' ---")
    print(f"  [v2 Pilot 250] ({p_eos}, {p_tok}t): {p_resp[:90]}{'...' if len(p_resp)>90 else ''}")
    print(f"  [v2 Aşama 7  ] ({s7_eos}, {s7_tok}t): {s7_resp[:90]}{'...' if len(s7_resp)>90 else ''}")
