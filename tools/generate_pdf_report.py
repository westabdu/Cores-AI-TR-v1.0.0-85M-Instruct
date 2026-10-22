"""
Cores-AI TR v2.0 — Profesyonel PDF Durum ve Değerlendirme Raporu Oluşturucu
v1 Base vs v2 Pilot vs v2 Aşama 7 Final Karşılaştırmalı Rapor
"""

import os
import sys
import json
import base64
import subprocess
from datetime import datetime

# Kök çalışma dizinine sabitle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def image_to_base64(path: str) -> str:
    """Resim dosyasını base64 data URI formatına çevirir."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = os.path.splitext(path)[1].replace(".", "").lower()
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def build_html_report():
    # 1. Verileri yükle
    with open("TESTLER/eval_v1_baseline.json", "r", encoding="utf-8") as f:
        v1_data = json.load(f)
    with open("TESTLER/eval_v2_sft.json", "r", encoding="utf-8") as f:
        pilot_data = json.load(f)
    with open("TESTLER/eval_v2_stage7.json", "r", encoding="utf-8") as f:
        st7_data = json.load(f)
    with open("TESTLER/benchmark_questions.json", "r", encoding="utf-8") as f:
        bm_data = json.load(f)

    # 2. Grafikleri base64 formatında al
    img_val_loss = image_to_base64("TESTLER/charts/val_loss_curve_complete.png")
    img_eos_comp = image_to_base64("TESTLER/charts/eos_comparison_3way.png")
    img_error_red = image_to_base64("TESTLER/charts/error_reduction_3way.png")
    img_dataset = image_to_base64("TESTLER/charts/dataset_composition.png")

    v1_res = {r["id"]: r for r in v1_data["results"]}
    p_res = {r["id"]: r for r in pilot_data["results"]}
    s7_res = {r["id"]: r for r in st7_data["results"]}

    # Kategori tablosu oluşturma
    cat_rows_html = ""
    categories = bm_data["categories"]
    for cid, cinfo in categories.items():
        v1_eos_cnt = sum(1 for r in v1_res.values() if r["category"] == cid and r["eos_found"])
        p_eos_cnt = sum(1 for r in p_res.values() if r["category"] == cid and r["eos_found"])
        s7_eos_cnt = sum(1 for r in s7_res.values() if r["category"] == cid and r["eos_found"])
        tot = cinfo["count"]
        p1 = v1_eos_cnt / tot * 100
        p_pilot = p_eos_cnt / tot * 100
        p_s7 = s7_eos_cnt / tot * 100
        diff = p_s7 - p1
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        diff_color = "#16a34a" if diff >= 0 else "#dc2626"

        cat_rows_html += f"""
        <tr>
            <td style="font-weight: bold; text-align: center;">{cid}</td>
            <td>{cinfo['name']}</td>
            <td style="text-align: center;">{tot}</td>
            <td style="text-align: center;">%{p1:.1f} ({v1_eos_cnt}/{tot})</td>
            <td style="text-align: center; color: #2563eb;">%{p_pilot:.1f} ({p_eos_cnt}/{tot})</td>
            <td style="text-align: center; font-weight: bold; color: #047857;">%{p_s7:.1f} ({s7_eos_cnt}/{tot})</td>
            <td style="text-align: center; font-weight: bold; color: {diff_color};">{diff_str}</td>
        </tr>
        """

    # Temsili karşılaştırma örnekleri (3 nesil kıyaslama)
    sample_pairs = [
        ("A01", "A", "Merhaba!", "Selamlama"),
        ("A02", "A", "Sen kimsin?", "Rol Tanıma"),
        ("B02", "B", "Türkiye'nin en büyük şehri", "Cümle Tamamlama"),
        ("C02", "C", "Gökyüzü hangi renktir?", "Faktüel Bilgi"),
        ("C05", "C", "Türkiye hangi kıtada yer alır?", "Faktüel Bilgi"),
        ("D03", "D", "Bana 3 meyve say.", "Talimat / Liste"),
        ("D20", "D", "Bana kısa bir teşekkür cümlesi yaz.", "Talimat"),
        ("E01", "E", "Şu metni özetle: 'Türkiye, Asya ve Avrupa kıtalarında...'", "Kısa Özetleme"),
        ("E05", "E", "Şu metni özetle: 'Sağlıklı beslenmek için sebze...'", "Kısa Özetleme"),
        ("F05", "F", "Şu konuyu değiştirme: Elma hakkında bir şey söyle.", "Tutarlılık / Odak"),
    ]

    samples_html = ""
    for qid, cat, prompt, desc in sample_pairs:
        v1_o = v1_res[qid]["output"].strip()
        p_o = p_res[qid]["assistant_response"].strip()
        s7_o = s7_res[qid]["assistant_response"].strip()

        v1_eos = "✅ EOS" if v1_res[qid]["eos_found"] else "❌ Kesildi (128t)"
        p_eos = "✅ EOS" if p_res[qid]["eos_found"] else "❌ Kesildi"
        s7_eos = "✅ EOS" if s7_res[qid]["eos_found"] else "❌ Kesildi"

        v1_tok = len(v1_o.split())
        p_tok = p_res[qid].get("generated_tokens", len(p_o.split()))
        s7_tok = s7_res[qid].get("generated_tokens", len(s7_o.split()))

        # HTML escape
        v1_clean = v1_o.replace("<", "&lt;").replace(">", "&gt;")
        p_clean = p_o.replace("<", "&lt;").replace(">", "&gt;")
        s7_clean = s7_o.replace("<", "&lt;").replace(">", "&gt;")

        samples_html += f"""
        <div class="sample-card">
            <div class="sample-header">
                <span class="sample-id">{qid}</span>
                <span class="sample-prompt">"{prompt}"</span>
                <span class="sample-badge">{desc}</span>
            </div>
            <div class="sample-body">
                <div class="sample-column v1-col">
                    <div class="column-title">v1 Base ({v1_eos})</div>
                    <div class="sample-text">{v1_clean}</div>
                </div>
                <div class="sample-column pilot-col">
                    <div class="column-title">v2 Pilot 250 ({p_eos} — {p_tok}t)</div>
                    <div class="sample-text">{p_clean}</div>
                </div>
                <div class="sample-column s7-col">
                    <div class="column-title">v2 Final 750 ({s7_eos} — {s7_tok}t)</div>
                    <div class="sample-text">{s7_clean}</div>
                </div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>Cores-AI TR v2.0 — Kapsamlı Durum ve Değerlendirme Raporu</title>
<style>
    @page {{
        size: A4;
        margin: 11mm 12mm 11mm 12mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        background: #ffffff;
        margin: 0;
        padding: 0;
        font-size: 8.8pt;
        line-height: 1.4;
    }}
    .page {{
        page-break-after: always;
        padding-bottom: 4px;
    }}
    .page:last-child {{
        page-break-after: avoid;
    }}
    
    /* Header */
    .report-header {{
        border-bottom: 2px solid #2563eb;
        padding-bottom: 8px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }}
    .report-title-box h1 {{
        font-size: 17pt;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 3px 0;
        letter-spacing: -0.4px;
    }}
    .report-title-box .subtitle {{
        font-size: 9.5pt;
        color: #475569;
        font-weight: 500;
    }}
    .header-badge {{
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        padding: 5px 10px;
        border-radius: 6px;
        font-size: 8pt;
        font-weight: bold;
        text-align: right;
    }}
    
    /* Headings */
    h2 {{
        font-size: 11.5pt;
        font-weight: 700;
        color: #0f172a;
        border-left: 3.5px solid #2563eb;
        padding-left: 7px;
        margin: 12px 0 8px 0;
        letter-spacing: -0.2px;
    }}
    h3 {{
        font-size: 9.5pt;
        font-weight: 600;
        color: #1e293b;
        margin: 8px 0 4px 0;
    }}
    p {{
        margin: 0 0 6px 0;
        color: #334155;
    }}
    
    /* Stat cards */
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 10px;
    }}
    .stat-card {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 10px;
        text-align: center;
    }}
    .stat-card .val {{
        font-size: 14pt;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 1px;
    }}
    .stat-card .val.green {{ color: #059669; }}
    .stat-card .val.blue {{ color: #2563eb; }}
    .stat-card .val.purple {{ color: #7c3aed; }}
    .stat-card .lbl {{
        font-size: 7.5pt;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}
    .stat-card .sub {{
        font-size: 7pt;
        color: #10b981;
        font-weight: bold;
        margin-top: 1px;
    }}
    
    /* Tables */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 8px 0 10px 0;
        font-size: 8pt;
    }}
    th {{
        background: #f1f5f9;
        color: #334155;
        font-weight: 700;
        text-align: left;
        padding: 5px 7px;
        border: 1px solid #cbd5e1;
    }}
    td {{
        padding: 4px 7px;
        border: 1px solid #e2e8f0;
        color: #334155;
    }}
    tr:nth-child(even) {{
        background: #f8fafc;
    }}
    
    /* Callouts */
    .callout {{
        background: #f0fdf4;
        border-left: 3.5px solid #16a34a;
        padding: 6px 10px;
        border-radius: 0 5px 5px 0;
        margin: 8px 0;
        font-size: 8.2pt;
        color: #14532d;
    }}
    .callout-title {{
        font-weight: bold;
        margin-bottom: 2px;
        color: #166534;
    }}
    .callout-blue {{
        background: #eff6ff;
        border-left: 3.5px solid #2563eb;
        color: #1e3a8a;
    }}
    .callout-blue .callout-title {{
        color: #1d4ed8;
    }}
    .callout-amber {{
        background: #fffbeb;
        border-left: 3.5px solid #d97706;
        color: #78350f;
    }}
    .callout-amber .callout-title {{
        color: #b45309;
    }}
    
    /* Chart container */
    .chart-row {{
        display: flex;
        gap: 10px;
        margin: 8px 0 10px 0;
        align-items: center;
    }}
    .chart-box {{
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 5px;
        text-align: center;
    }}
    .chart-box img {{
        max-width: 100%;
        height: auto;
        border-radius: 4px;
        display: block;
        margin: 0 auto;
    }}
    .chart-caption {{
        font-size: 7.5pt;
        color: #64748b;
        margin-top: 3px;
        font-weight: 600;
    }}
    
    /* Sample cards */
    .sample-card {{
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        margin-bottom: 6px;
        overflow: hidden;
        font-size: 7.5pt;
        background: #ffffff;
    }}
    .sample-header {{
        background: #f8fafc;
        padding: 3px 6px;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .sample-id {{
        font-weight: 800;
        color: #2563eb;
        background: #dbeafe;
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 7pt;
    }}
    .sample-prompt {{
        font-weight: 700;
        color: #0f172a;
        flex: 1;
    }}
    .sample-badge {{
        font-size: 6.5pt;
        background: #e2e8f0;
        color: #475569;
        padding: 1px 5px;
        border-radius: 3px;
    }}
    .sample-body {{
        display: flex;
    }}
    .sample-column {{
        flex: 1;
        padding: 4px 6px;
    }}
    .v1-col {{
        border-right: 1px solid #e2e8f0;
        background: #fffafa;
    }}
    .pilot-col {{
        border-right: 1px solid #e2e8f0;
        background: #f8fafc;
    }}
    .s7-col {{
        background: #f0fdf4;
    }}
    .column-title {{
        font-size: 6.5pt;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 2px;
    }}
    .v1-col .column-title {{ color: #dc2626; }}
    .pilot-col .column-title {{ color: #2563eb; }}
    .s7-col .column-title {{ color: #16a34a; }}
    .sample-text {{
        font-family: Consolas, "Courier New", monospace;
        font-size: 7pt;
        white-space: pre-wrap;
        color: #334155;
        max-height: 48px;
        overflow: hidden;
    }}
</style>
</head>
<body>

<!-- ==================== SAYFA 1 ==================== -->
<div class="page">
    <div class="report-header">
        <div class="report-title-box">
            <h1>🇹🇷 Cores-AI TR v2.0 — Kapsamlı Durum ve Değerlendirme Raporu</h1>
            <div class="subtitle">v1.0.0 Base (Pre-training) → v2.0.0 SFT (Diyalog & Asistan) Tam Süreç Analizi</div>
        </div>
        <div class="header-badge">
            SFT v2.0 Tamamlandı<br>
            <span style="font-size: 7.5pt; font-weight: normal; color: #475569;">16 Ekim 2026 | Resmileştirildi</span>
        </div>
    </div>

    <!-- ÖZET KARTLAR -->
    <div class="stat-grid">
        <div class="stat-card">
            <div class="val blue">81.17M</div>
            <div class="lbl">Parametre Sayısı</div>
            <div class="sub">10 Katman, 10 Kafa, 640 Embd</div>
        </div>
        <div class="stat-card">
            <div class="val green">%93.3</div>
            <div class="lbl">EOS Vurma (Susma)</div>
            <div class="sub">+%36.6 İyileşme (v1: %56.7)</div>
        </div>
        <div class="stat-card">
            <div class="val green">%0.0</div>
            <div class="lbl">Vikipedi / Tablo Kodu</div>
            <div class="sub">%100 Temizlendi (v1: %54.7)</div>
        </div>
        <div class="stat-card">
            <div class="val purple">3.9976</div>
            <div class="lbl">Final Val Loss</div>
            <div class="sub">&lt; 4.0 Hedefine Ulaşıldı (750 Adım)</div>
        </div>
    </div>

    <div class="callout callout-blue">
        <div class="callout-title">📌 Yönetici Özeti (Büyük v2.0 SFT Güncellemesi)</div>
        Bu rapor; normalde yalnızca ham bir ön eğitim modeli (<strong>v1 Base Model</strong>) olarak kalması planlanan Cores-AI TR projesinin, radikal bir kararla Supervised Fine-Tuning (SFT) sürecine alınarak diyalog kurabilen, kullanıcıya saygılı ve işi bitince susmayı bilen bir Türkçe asistan modeline (<strong>v2 SFT Model</strong>) dönüştürülme sürecini belgeler. 250 adımlık pilot ve 500 adımlık devam eğitimiyle <strong>toplam 750 adım</strong> tamamlanmış; validation loss <strong>4.6248'den 3.9976'ya</strong> düşürülmüş ve Vikipedi kirliliği <strong>%54.7'den tam olarak %0.0'a</strong> indirilmiştir. Nihai model <code>out-cores-85m-sft-v2/</code> olarak resmileştirilmiştir.
    </div>

    <h2>1. Model Mimarisi ve Teknik Yapılandırma</h2>
    <table style="margin-bottom: 6px;">
        <tr>
            <th width="24%">Bileşen</th>
            <th width="36%">Teknik Değer</th>
            <th width="40%">Açıklama / Tasarım Kararı</th>
        </tr>
        <tr>
            <td><strong>Temel Mimari</strong></td>
            <td>NanoGPT Decoder-Only Transformer</td>
            <td>10 Katman (n_layer), 10 Dikkat Kafası (n_head), 640 Embedding boyutu</td>
        </tr>
        <tr>
            <td><strong>Bağlam Penceresi</strong></td>
            <td>512 Token (block_size)</td>
            <td>Tüm padding ve dikkat maskeleri mutlak 512 sınırına göre hizalanmıştır</td>
        </tr>
        <tr>
            <td><strong>Tokenizer</strong></td>
            <td><code>tokenizer_tr_v2.json</code> (50.000 BPE)</td>
            <td><code>[UNK]=0</code>, <code>[PAD]=1</code>, <code>[BOS]=2</code>, <code>[EOS]=3</code> (Sözlük korunmuştur)</td>
        </tr>
        <tr>
            <td><strong>Diyalog Şablonu</strong></td>
            <td><code>[BOS]Kullanıcı: ...\\nAsistan: ...[EOS]</code></td>
            <td>Yeni özel token eklenmeden, mevcut BPE sözlüğü ile kusursuz uyum sağlandı</td>
        </tr>
        <tr>
            <td><strong>Loss Masking</strong></td>
            <td><code>ignore_index = -1</code></td>
            <td>Kullanıcı sorusu ve şablon gradyandan maskelenir; yalnızca asistan yanıtı eğitilir</td>
        </tr>
    </table>

    <h2>2. SFT Veri Kümesi Hazırlığı (Aşama 3 & 5B)</h2>
    <p>Modelin Türkçe gramer tabanı v1 pre-training ile önceden oturduğundan, modele yeni dil öğretmek yerine diyalog formatında asistan rolü kazandırılması hedeflenmiştir. Üç açık lisanslı kaynaktan sızıntı korumalı <strong>8.936 temiz örnek</strong> filtrelenmiştir:</p>

    <div class="chart-row">
        <div class="chart-box" style="flex: 1.1;">
            <img src="{img_dataset}" alt="Veri Dağılımı">
            <div class="chart-caption">Şekil 1: SFT Veri Dağılımı (Train: 8.043 [%90], Val: 893 [%10])</div>
        </div>
        <div class="chart-box" style="flex: 0.9; text-align: left; padding: 8px 10px; font-size: 7.5pt;">
            <strong style="color: #0f172a; font-size: 8pt;">Veri Filtreleme Prensipleri:</strong>
            <ul style="margin: 3px 0; padding-left: 14px; color: #475569;">
                <li><strong>Sıfır Benchmark Sızıntısı:</strong> 150 benchmark sorusunun tamamı eğitim verisinden izole edildi.</li>
                <li><strong>Uzunluk Denetimi:</strong> 512 token sınırını aşan veya boş yanıtlar ayıklandı.</li>
                <li><strong>Kullanıcı/Asistan Normalizasyonu:</strong> Farklı formatlardaki talimatlar tek tip şablona dönüştürüldü.</li>
                <li><strong>Veri Bütünlüğü:</strong> Train ve Val kümeleri arasında çakışma olmadığı hash kontrolüyle doğrulandı.</li>
            </ul>
        </div>
    </div>
</div>

<!-- ==================== SAYFA 2 ==================== -->
<div class="page">
    <h2>3. Eğitim Dinamikleri ve Kayıp Eğrisi (Aşama 5C & Aşama 7)</h2>
    <p>Eğitim iki fazda koşturulmuştur: 250 adımlık Pilot (lr: 2e-5) ve 500 adımlık Aşama 7 Devam Eğitimi (lr: 1e-5). Toplam 750 adım sonunda validation loss düzenli olarak gerilemiştir (4.6248 → 3.9976).</p>

    <div class="chart-row">
        <div class="chart-box" style="flex: 1;">
            <img src="{img_val_loss}" alt="Complete Val Loss Curve">
            <div class="chart-caption">Şekil 2: 750 Adımlık Kesintisiz Validation Loss Eğrisi (4.6248 → 3.9976)</div>
        </div>
        <div class="chart-box" style="flex: 1;">
            <img src="{img_error_red}" alt="3-Way Error Reduction">
            <div class="chart-caption">Şekil 3: Üç Nesil Hata ve Kirlilik Azalması (v1 vs Pilot vs Aşama 7)</div>
        </div>
    </div>

    <div class="callout">
        <div class="callout-title">🎯 Sıfır Overfitting — Başarılı Doyum Noktası</div>
        Validation loss her değerlendirme adımında düşmeye devam etmiş ve 750. adımda <strong>3.9976</strong> ile en iyi kontrol noktası kaydedilmiştir. İkinci fazda öğrenme oranının 1e-5'e çekilmesi, modelin önceki ağırlıkları bozmadan (catastrophic forgetting yaşamadan) diyalog kalitesini derinleştirmesini sağlamıştır.
    </div>

    <h2>4. Üç Yönlü Karşılaştırmalı Benchmark Değerlendirmesi</h2>
    <p>150 soruluk bağımsız test seti üzerinde aynı değerlendirme ayarlarıyla (temp: 0.7, top-k: 50, max_new_tokens: 128, seed: 42, CPU) üç model test edilmiştir:</p>

    <table style="margin-bottom: 6px;">
        <thead>
            <tr>
                <th width="28%">Ölçülen Metrik</th>
                <th width="24%" style="text-align: center;">v1 Baseline (Ham)</th>
                <th width="24%" style="text-align: center;">v2 Pilot (250 Adım)</th>
                <th width="24%" style="text-align: center;">v2 Aşama 7 Final (750 Adım)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>EOS Vurma (Doğru Durma)</strong></td>
                <td style="text-align: center;">85 / 150 (%56.7)</td>
                <td style="text-align: center; color: #2563eb;">142 / 150 (%94.7)</td>
                <td style="text-align: center; font-weight: bold; color: #047857;">140 / 150 (%93.3)</td>
            </tr>
            <tr>
                <td><strong>Vikipedi / Tablo / HTML Kirliliği</strong></td>
                <td style="text-align: center; color: #dc2626;">82 / 150 (%54.7)</td>
                <td style="text-align: center;">2 / 150 (%1.3)</td>
                <td style="text-align: center; font-weight: bold; color: #047857;">0 / 150 (%0.0 — Sıfırlandı)</td>
            </tr>
            <tr>
                <td><strong>Ağır Tekrar Döngüsü</strong></td>
                <td style="text-align: center; color: #dc2626;">40 / 150 (%26.7)</td>
                <td style="text-align: center;">15 / 150 (%10.0)</td>
                <td style="text-align: center; font-weight: bold; color: #047857;">14 / 150 (%9.3)</td>
            </tr>
            <tr>
                <td><strong>128 Token Dolup Kesilme (Cut)</strong></td>
                <td style="text-align: center; color: #dc2626;">65 / 150 (%43.3)</td>
                <td style="text-align: center;">8 / 150 (%5.3)</td>
                <td style="text-align: center; font-weight: bold; color: #047857;">10 / 150 (%6.7)</td>
            </tr>
            <tr>
                <td><strong>Ortalama Üretilen Token Sayısı</strong></td>
                <td style="text-align: center;">~85.0 token (susamıyor)</td>
                <td style="text-align: center;">25.9 token (özet/net)</td>
                <td style="text-align: center; font-weight: bold;">24.7 token (özet/net)</td>
            </tr>
        </tbody>
    </table>

    <table style="margin-bottom: 6px;">
        <thead>
            <tr>
                <th width="6%">Kod</th>
                <th width="32%">Kategori Adı</th>
                <th width="10%" style="text-align: center;">Soru</th>
                <th width="14%" style="text-align: center;">v1 Base EOS</th>
                <th width="14%" style="text-align: center;">v2 Pilot EOS</th>
                <th width="14%" style="text-align: center;">v2 Final EOS</th>
                <th width="10%" style="text-align: center;">Net Fark</th>
            </tr>
        </thead>
        <tbody>
            {cat_rows_html}
            <tr style="background: #f1f5f9; font-weight: bold;">
                <td colspan="2" style="text-align: right;">GENEL ORTALAMA:</td>
                <td style="text-align: center;">150</td>
                <td style="text-align: center;">%56.7 (85)</td>
                <td style="text-align: center; color: #2563eb;">%94.7 (142)</td>
                <td style="text-align: center; color: #047857;">%93.3 (140)</td>
                <td style="text-align: center; color: #16a34a;">+36.6%</td>
            </tr>
        </tbody>
    </table>

    <div class="chart-box" style="margin-top: 4px;">
        <img src="{img_eos_comp}" alt="3-Way EOS Comparison" style="max-height: 180px;">
        <div class="chart-caption">Şekil 4: 7 Kategori Bazında 3 Nesil EOS Vurma (Susma) Oranları Karşılaştırması</div>
    </div>
</div>

<!-- ==================== SAYFA 3 ==================== -->
<div class="page">
    <h2>5. Temsilî Soru-Cevap Çıktı Karşılaştırması</h2>
    <p>Aşağıdaki 10 temsilî soru, modelin ham ansiklopedi metin tamamlayıcısı kimliğinden çıkarak diyalog kurabilen bir asistan kimliğine nasıl evrildiğini açıkça ortaya koymaktadır:</p>

    <div class="samples-container">
        {samples_html}
    </div>
</div>

<!-- ==================== SAYFA 4 ==================== -->
<div class="page">
    <h2>6. Gerçekçi Model Değerlendirmesi ve Bilinen Sınırlamalar</h2>
    <div class="callout callout-amber">
        <div class="callout-title">⚠️ 81.17M Parametre Kısıtları — Şeffaf Değerlendirme</div>
        Cores-AI TR, tüketici donanımlarında CPU üzerinde çalışabilen 81.17M parametreli ultra hafif bir modeldir. 7B/70B modellerle kıyaslanamaz:
        <ul style="margin: 3px 0; padding-left: 14px; font-size: 7.8pt;">
            <li><strong>Faktüel Doğruluk ve Halüsinasyon:</strong> Model genel kültür, coğrafya veya fen sorularında güvenilir bilgi kaynağı DEĞİLDİR. Olgusal sorularda gerçeğe aykırı metinler üretebilir.</li>
            <li><strong>Yankılama (Echoing):</strong> "Bana 3 meyve say" veya "Bana tek kelime söyle" gibi kısıt içeren soyut emirlerde sorunun kendisini kullanıcıya geri yansıtabilmektedir.</li>
            <li><strong>Asıl Başarı:</strong> Modelin en büyük başarısı, format kirliliğini (%54.7'den %0.0'a) tamamen yok etmesi, diyalog başlatması ve nazikçe susmayı öğrenmesidir.</li>
        </ul>
    </div>

    <h2>7. Dosya Bütünlüğü ve Checkpoint Güvenlik Tablosu</h2>
    <table style="margin-bottom: 8px;">
        <thead>
            <tr>
                <th width="26%">Kontrol Noktası / Dosya</th>
                <th width="44%">SHA-256 Hash Değeri</th>
                <th width="15%" style="text-align: center;">Boyut</th>
                <th width="15%" style="text-align: center;">Durum</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>v1 Base (Pre-training)</strong><br><code>out-cores-85m-test/ckpt.pt</code></td>
                <td><code style="font-size: 6.5pt;">e88f284fdfb9ddc48075a97e47f509859319590df312a4226d9b355c63347e51</code></td>
                <td style="text-align: center;">977.9 MB</td>
                <td style="text-align: center; color: #16a34a; font-weight: bold;">🔒 KORUNDU</td>
            </tr>
            <tr>
                <td><strong>v2 SFT Pilot (250 Adım)</strong><br><code>TESTLER/temp_sft_pilot/ckpt_best.pt</code></td>
                <td><code style="font-size: 6.5pt;">106343294c6af89f3bc9201b6a51a78c527b2657cd3cdfc4d2942c03e0d8bb6a</code></td>
                <td style="text-align: center;">977.9 MB</td>
                <td style="text-align: center; color: #16a34a; font-weight: bold;">🔒 ARŞİVLENDİ</td>
            </tr>
            <tr>
                <td><strong>v2 SFT Aşama 7 (Devam)</strong><br><code>TESTLER/temp_sft_stage7/ckpt_best.pt</code></td>
                <td><code style="font-size: 6.5pt;">3fc057b04e3b949291b5dc249ba420f511d868cf88b2a4da29c884315cb71618</code></td>
                <td style="text-align: center;">977.9 MB</td>
                <td style="text-align: center; color: #16a34a; font-weight: bold;">✅ DOĞRULANDI</td>
            </tr>
            <tr>
                <td><strong>v2 RESMİ YAYIN SÜRÜMÜ</strong><br><code>out-cores-85m-sft-v2/ckpt.pt</code></td>
                <td><code style="font-size: 6.5pt;">3fc057b04e3b949291b5dc249ba420f511d868cf88b2a4da29c884315cb71618</code></td>
                <td style="text-align: center;">977.9 MB</td>
                <td style="text-align: center; color: #047857; font-weight: bold;">🚀 AKTİF SÜRÜM</td>
            </tr>
        </tbody>
    </table>

    <h2>8. GitHub & Hugging Face Dağıtım Mimarisi</h2>
    <table style="margin-bottom: 8px;">
        <tr>
            <th width="20%">Platform</th>
            <th width="40%">Yüklenecek Dosyalar</th>
            <th width="40%">Kural / Yöntem</th>
        </tr>
        <tr>
            <td><strong>GitHub</strong></td>
            <td>Tüm Python kodları (<code>model.py</code>, <code>train_sft.py</code>, <code>sample_sft.py</code>, <code>tools/</code>), <code>README.md</code>, test JSON/grafik çıktıları</td>
            <td><code>.gitignore</code> ile <code>*.pt</code> ve <code>out-*/</code> klasörleri otomatik filtrelenir (&lt;100 MB kuralı).</td>
        </tr>
        <tr>
            <td><strong>Hugging Face Hub</strong><br><code>aapo33/cores-ai-tr-v1.0.0-85m</code></td>
            <td><code>out-cores-85m-sft-v2/ckpt.pt</code> (~978 MB)<br><code>out-cores-85m-test/ckpt.pt</code> (~978 MB)<br><code>tokenizer_tr_v2.json</code></td>
            <td><code>huggingface-cli upload</code> veya <code>huggingface_hub</code> Python kütüphanesi ile doğrudan model reposuna yüklenir.</td>
        </tr>
    </table>

    <div class="callout callout-blue" style="margin-top: 6px;">
        <div class="callout-title">💬 Modeli Çalıştırma (Hızlı Başlangıç)</div>
        Depo klonlandıktan ve ağırlıklar indirildikten sonra çok turlu interaktif sohbeti başlatmak için:
        <br><code>python sample_sft.py --device cpu</code>
    </div>

    <div style="margin-top: 20px; padding-top: 8px; border-top: 1px solid #cbd5e1; display: flex; justify-content: space-between; font-size: 7.5pt; color: #94a3b8;">
        <span>Cores-AI TR v2.0 Final Raporu — DeepMind Antigravity Pair-Programming</span>
        <span>Tarih: 16 Ekim 2026 | Durum: Tamamlandı ve Resmileştirildi</span>
    </div>
</div>

</body>
</html>
"""
    return html


def main():
    output_html = "TESTLER/cores_v2_durum_raporu.html"
    output_pdf = "TESTLER/Cores_AI_TR_v2_Durum_Raporu.pdf"
    root_pdf = "Cores_AI_TR_v2_Durum_Raporu.pdf"

    print("📄 HTML raporu derleniyor...")
    html_content = build_html_report()
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   ✅ HTML kaydedildi: {output_html}")

    print("\n🖨️ Microsoft Edge headless ile yüksek kaliteli PDF derleniyor...")
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    abs_html = os.path.abspath(output_html)
    abs_pdf = os.path.abspath(output_pdf)
    abs_root_pdf = os.path.abspath(root_pdf)

    args = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html.replace(os.sep, '/')}"
    ]

    res = subprocess.run(args, capture_output=True, text=True)
    if os.path.exists(abs_pdf) and os.path.getsize(abs_pdf) > 1000:
        pdf_size = os.path.getsize(abs_pdf) / 1024
        print(f"   ✅ PDF başarıyla oluşturuldu: {output_pdf} ({pdf_size:.1f} KB)")
        
        # Kök dizine de kopyala (kullanıcı kolaylığı için)
        import shutil
        shutil.copyfile(abs_pdf, abs_root_pdf)
        print(f"   ✅ Kök dizine kopyalandı: {root_pdf}")

        if os.path.exists(output_html):
            os.remove(output_html)
            print(f"   🧹 Geçici HTML temizlendi: {output_html}")
    else:
        print(f"❌ PDF oluşturma hatası: {res.stderr}")


if __name__ == "__main__":
    main()
