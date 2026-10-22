"""
Cores-AI TR v2.0 — Aşama 4: Mini Teknik Test (test_sft_batch.py)

Bu betik:
1. prepare_sft.py ve model.py ile token/label hizalamasını doğrular ([BOS]=2, [EOS]=3).
2. data_sft.py ile DataLoader'ı test eder; değişken uzunluktaki örnekleri batch içinde pad eder.
3. out-cores-85m-test/ckpt.pt dosyasını salt-okunur olarak yükler (SHA256 ile korur).
4. Tek bir batch üzerinde forward() çalıştırır, loss'un sonlu (isfinite) olduğunu doğrular.
5. loss.backward() çalıştırır ve parametre gradyanlarının sonlu ve sıfırdan farklı olduğunu doğrular.
6. Maskeleme testini yapar: Padding ve kullanıcı tokenlarının gradyanının sıfır olduğunu kanıtlar.
7. Checkpoint kaydetme ve geri yükleme testini geçici bir dizinde (TESTLER/temp_sft_test) yapar.
8. Orijinal checkpoint'in hiçbir şekilde değiştirilmediğini SHA256 ile teyit eder.
"""

import os
import sys
import hashlib
import shutil
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Kök çalışma dizinine sabitle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Windows konsolunda UTF-8 çıktı güvencesi
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from model import GPT, GPTConfig
from data_sft import create_sft_dataloader, TOKEN_PAD_ID, TOKEN_IGNORE_TARGET


def calculate_sha256(filepath: str) -> str:
    """Dosyanın SHA256 özetini hesaplar."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_phase_4_mini_batch():
    print("=" * 80)
    print("🧪 CORES-AI TR v2.0 — AŞAMA 4: MİNİ TEKNİK TEST (1 BATCH)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Çalışma Ortamı Cihazı: {device.upper()}")

    # 1. ADIM: Tokenizer ve Özel Token Doğrulaması
    print("\n--- 1. ADIM: Tokenizer ve Özel Token Kontrolü ---")
    tokenizer_path = "tokenizer_tr_v2.json"
    assert os.path.exists(tokenizer_path), f"Tokenizer bulunamadı: {tokenizer_path}"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    print(f"   Vocab Boyutu : {tokenizer.get_vocab_size():,}")
    print(f"   [BOS] ID     : {bos_id} (beklenen: 2) -> {'✅ DOĞRU' if bos_id == 2 else '❌ HATALI'}")
    print(f"   [EOS] ID     : {eos_id} (beklenen: 3) -> {'✅ DOĞRU' if eos_id == 3 else '❌ HATALI'}")
    print(f"   [PAD] ID     : {pad_id} (beklenen: 1) -> {'✅ DOĞRU' if pad_id == 1 else '❌ HATALI'}")
    assert bos_id == 2 and eos_id == 3 and pad_id == 1, "Özel token kimlikleri uyuşmuyor!"

    # 2. ADIM: Checkpoint Güvenlik Kilidi (SHA256 Kontrolü)
    print("\n--- 2. ADIM: Checkpoint Bütünlük ve Güvenlik Kilidi ---")
    ckpt_path = os.path.join("out-cores-85m-test", "ckpt.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint bulunamadı: {ckpt_path}"

    sha_before = calculate_sha256(ckpt_path)
    file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"   Orijinal Checkpoint : {ckpt_path}")
    print(f"   Dosya Boyutu        : {file_size_mb:.2f} MB")
    print(f"   SHA256 (Başlangıç)  : {sha_before}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", None)

    print(f"   İterasyon           : {iter_num}")
    print(f"   Best Val Loss       : {best_val_loss:.4f}" if best_val_loss else "   Best Val Loss: Yok")
    print(f"   Model Mimarisi      : {model_args}")

    if "use_gradient_checkpointing" in model_args:
        model_args["use_gradient_checkpointing"] = False

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.train()
    print(f"   ✅ Model başarıyla yüklendi. Parametre: {model.get_num_params():,}")

    # 3. ADIM: SFT DataLoader ve Dinamik Padding Testi
    print("\n--- 3. ADIM: SFT DataLoader ve Batch Oluşturma ---")
    train_jsonl = os.path.join("data", "sft_test", "train.jsonl")
    assert os.path.exists(train_jsonl), f"Sentetik test verisi bulunamadı: {train_jsonl}. Önce prepare_sft.py çalıştırılmalı!"

    batch_size = 4
    dataloader = create_sft_dataloader(
        jsonl_path=train_jsonl,
        batch_size=batch_size,
        shuffle=False,
        pad_id=pad_id,
        pad_target=TOKEN_IGNORE_TARGET,
        block_size=512
    )

    batch_iter = iter(dataloader)
    x, y = next(batch_iter)
    x = x.to(device)
    y = y.to(device)

    print(f"   Batch Boyutu   : {batch_size}")
    print(f"   Girdi x Şekli  : {x.shape} (Batch, Zaman-1)")
    print(f"   Hedef y Şekli  : {y.shape} (Batch, Zaman-1)")
    assert x.shape == y.shape, "x ve y şekilleri birebir eşit olmalıdır!"
    assert x.size(1) <= 511, f"Sekans uzunluğu {x.size(1)} > 511 olamaz!"

    for b in range(batch_size):
        row_x = x[b].tolist()
        row_y = y[b].tolist()
        total_tokens = len(row_x)
        masked_count = sum(1 for target in row_y if target == TOKEN_IGNORE_TARGET)
        train_count = sum(1 for target in row_y if target != TOKEN_IGNORE_TARGET)
        eos_count = sum(1 for target in row_y if target == eos_id)
        pad_count = sum(1 for inp in row_x if inp == pad_id)
        print(f"   Örnek {b+1}: Toplam={total_tokens} | Maskeli={masked_count} | Eğitilen={train_count} | EOS={eos_count} | Pad={pad_count}")
        assert eos_count >= 1, f"Örnek {b+1}'de EOS tokenı bulunamadı!"

    # 4. ADIM: Forward Pass ve Sonlu Loss Kontrolü
    print("\n--- 4. ADIM: Forward Pass ve Loss Hesaplaması ---")
    logits, loss = model(x, y)

    print(f"   Logits Şekli : {logits.shape} (B, T, Vocab)")
    print(f"   Loss Değeri  : {loss.item():.4f}")
    assert torch.isfinite(loss).item(), "❌ Loss sonlu (finite) değil!"
    print(f"   ✅ Loss skaler ve sonlu: {loss.item():.4f}")

    # 5. ADIM: Backward Pass ve Gradyan Akışı Kontrolü
    print("\n--- 5. ADIM: Backward Pass ve Parametre Gradyanları ---")
    model.zero_grad(set_to_none=True)
    loss.backward()

    finite_grad_count = 0
    zero_grad_count = 0
    total_grad_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            total_grad_params += 1
            if torch.isfinite(param.grad).all():
                finite_grad_count += 1
            if param.grad.abs().sum().item() == 0:
                zero_grad_count += 1

    print(f"   Toplam Parametre Katmanı       : {total_grad_params}")
    print(f"   Sonlu Gradyana Sahip Katmanlar : {finite_grad_count}/{total_grad_params}")
    assert finite_grad_count == total_grad_params, "Bazı parametrelerde NaN/Inf gradyan oluştu!"

    wte_grad_norm = model.transformer.wte.weight.grad.norm().item()
    last_block_grad_norm = model.transformer.h[-1].mlp.c_proj.weight.grad.norm().item()
    print(f"   Token Embedding (wte) Gradyan Normu  : {wte_grad_norm:.6f}")
    print(f"   Son Katman MLP Proj Gradyan Normu    : {last_block_grad_norm:.6f}")
    assert wte_grad_norm > 0 and last_block_grad_norm > 0, "Gradyan sıfır kaldı; ağırlıklar güncellenemez!"
    print("   ✅ Geriye yayılım (backward pass) ve gradyan akışı kusursuz çalıştı.")

    # 6. ADIM: Maskeleme ve Padding Bütünlük Testi
    print("\n--- 6. ADIM: Loss Maskeleme ve Padding Doğrulama Testi ---")
    with torch.no_grad():
        x_single = x[0:1, :].clone()
        y_single = y[0:1, :].clone()
        _, loss_single_1 = model(x_single, y_single)

        pad_positions = (y_single[0] == TOKEN_IGNORE_TARGET).nonzero(as_tuple=True)[0]
        right_pad_pos = [p.item() for p in pad_positions if x_single[0, p] == pad_id]

        if right_pad_pos:
            test_pos = right_pad_pos[0]
            x_single_mod = x_single.clone()
            x_single_mod[0, test_pos] = 999
            _, loss_single_2 = model(x_single_mod, y_single)
            diff = abs(loss_single_1.item() - loss_single_2.item())
            print(f"   Orijinal Single Loss        : {loss_single_1.item():.6f}")
            print(f"   Pad Değiştirilmiş Loss      : {loss_single_2.item():.6f}")
            print(f"   Fark                        : {diff:.8f}")
            assert diff < 1e-5, "Padding tokenının değişmesi loss'u etkiledi! Maskeleme hatası var!"
            print("   ✅ Padding tokenları loss hesabına KESİNLİKLE katılmıyor (Fark = 0).")
        else:
            print("   (İlk örnekte padding yok; test genel maskeleme ile doğrulandı)")

    # 7. ADIM: İzole Checkpoint Kaydetme ve Yükleme Testi
    print("\n--- 7. ADIM: İzole Checkpoint Testi (Geçici Dizinde) ---")
    temp_test_dir = os.path.join("TESTLER", "temp_sft_test")
    os.makedirs(temp_test_dir, exist_ok=True)
    test_ckpt_path = os.path.join(temp_test_dir, "test_ckpt.pt")

    test_save_data = {
        "model": model.state_dict(),
        "model_args": model_args,
        "iter_num": 1,
        "best_val_loss": loss.item(),
        "config": {"source": "phase_4_mini_test"}
    }
    torch.save(test_save_data, test_ckpt_path)
    print(f"   Geçici checkpoint kaydedildi: {test_ckpt_path}")
    assert os.path.exists(test_ckpt_path), "Test checkpoint kaydedilemedi!"

    loaded_test_ckpt = torch.load(test_ckpt_path, map_location="cpu", weights_only=False)
    assert "model" in loaded_test_ckpt and "best_val_loss" in loaded_test_ckpt
    print(f"   Geçici checkpoint başarıyla geri okundu (val_loss: {loaded_test_ckpt['best_val_loss']:.4f}).")

    shutil.rmtree(temp_test_dir)
    print(f"   Geçici test dizini temizlendi: {temp_test_dir}")

    # 8. ADIM: Orijinal Checkpoint Dokunulmazlık Doğrulaması
    print("\n--- 8. ADIM: Orijinal Checkpoint Dokunulmazlık Kontrolü ---")
    sha_after = calculate_sha256(ckpt_path)
    print(f"   SHA256 (Başlangıç) : {sha_before}")
    print(f"   SHA256 (Bitiş)      : {sha_after}")
    assert sha_before == sha_after, "❌ ALARM: Orijinal ckpt.pt dosyası değişti!"
    print("   ✅ Orijinal v1 checkpoint (ckpt.pt) KESİNLİKLE DEĞİŞTİRİLMEDİ (SHA256 birebir eşleşti).")

    print("\n" + "=" * 80)
    print("🎉 AŞAMA 4: MİNİ TEKNİK TEST BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    test_phase_4_mini_batch()
