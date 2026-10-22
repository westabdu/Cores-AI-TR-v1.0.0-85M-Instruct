"""
Cores-AI TR — Kategori Bazlı Detaylı Kalite İncelemesi
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
cats = bm_data['categories']

wiki_patterns = [r'\|\-', r'colspan', r'style=', r'\{\|\s*class', r'Kaynakça', r'Dış bağlantılar', r'Kategori:', r'==\s*Tarihçe\s*==']

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
            if c1 == c2 == c3 and len(c1) > 3:
                return True
    return False

print("=== EOS İSTATİSTİKLERİ ===")
for cid, cinfo in cats.items():
    v1_c_eos = sum(1 for r in v1_res.values() if r['category'] == cid and r['eos_found'])
    v2_c_eos = sum(1 for r in v2_res.values() if r['category'] == cid and r['eos_found'])
    tot = cinfo['count']
    p1 = v1_c_eos / tot * 100
    p2 = v2_c_eos / tot * 100
    print(f"{cid} | {cinfo['name']}: v1={v1_c_eos}/{tot} ({p1:.1f}%) -> v2={v2_c_eos}/{tot} ({p2:.1f}%) | fark=+{p2-p1:.1f}%")

total_v1_eos = sum(1 for r in v1_res.values() if r['eos_found'])
total_v2_eos = sum(1 for r in v2_res.values() if r['eos_found'])
print(f"TOPLAM EOS: v1={total_v1_eos}/150 ({total_v1_eos/1.5:.1f}%) -> v2={total_v2_eos}/150 ({total_v2_eos/1.5:.1f}%)")

v1_wiki = sum(1 for r in v1_res.values() if check_wiki(r['output']))
v2_wiki = sum(1 for r in v2_res.values() if check_wiki(r['assistant_response']))
print(f"VİKİPEDİ KODU: v1={v1_wiki}/150 ({v1_wiki/1.5:.1f}%) -> v2={v2_wiki}/150 ({v2_wiki/1.5:.1f}%)")

v1_rep = sum(1 for r in v1_res.values() if check_repetition(r['output']))
v2_rep = sum(1 for r in v2_res.values() if check_repetition(r['assistant_response']))
print(f"AĞIR TEKRAR DÖNGÜSÜ: v1={v1_rep}/150 ({v1_rep/1.5:.1f}%) -> v2={v2_rep}/150 ({v2_rep/1.5:.1f}%)")

print("\n=== KATEGORİ A İNCELEMESİ (Selamlama ve Rol Tanıma) ===")
for i in range(1, 11):
    qid = f"A{i:02d}"
    prompt = v1_res[qid]['prompt']
    v1_o = v1_res[qid]['output'].replace('\n', ' ')[:60]
    v2_o = v2_res[qid]['assistant_response'].replace('\n', ' ')[:60]
    print(f"[{qid}] '{prompt}'")
    print(f"   v1: {v1_o}")
    print(f"   v2: {v2_o}")

print("\n=== KATEGORİ C İNCELEMESİ (Faktüel Soru-Cevap) ===")
for i in range(1, 11):
    qid = f"C{i:02d}"
    prompt = v1_res[qid]['prompt']
    v1_o = v1_res[qid]['output'].replace('\n', ' ')[:60]
    v2_o = v2_res[qid]['assistant_response'].replace('\n', ' ')[:60]
    print(f"[{qid}] '{prompt}'")
    print(f"   v1: {v1_o}")
    print(f"   v2: {v2_o}")

print("\n=== KATEGORİ D İNCELEMESİ (Talimat Takibi) ===")
for i in range(1, 11):
    qid = f"D{i:02d}"
    prompt = v1_res[qid]['prompt']
    v1_o = v1_res[qid]['output'].replace('\n', ' ')[:60]
    v2_o = v2_res[qid]['assistant_response'].replace('\n', ' ')[:60]
    print(f"[{qid}] '{prompt}'")
    print(f"   v1: {v1_o}")
    print(f"   v2: {v2_o}")
