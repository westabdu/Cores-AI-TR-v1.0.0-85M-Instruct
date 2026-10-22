"""
Cores-AI TR — Hugging Face Model Ağırlıkları Yükleme Aracı (upload_to_hf.py)
GitHub'ın 100 MB limiti nedeniyle 978 MB'lık checkpoint dosyalarını doğrudan Hugging Face Model Hub'a yükler.
"""

import os
import sys
import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Cores-AI TR Model Ağırlıklarını Hugging Face'e Yükleme Aracı")
    parser.add_argument("--repo", type=str, default="aapo33/cores-ai-tr-v1.0.0-85m",
                        help="Hugging Face repo id (örn: aapo33/cores-ai-tr-v1.0.0-85m)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face Write Token (https://huggingface.co/settings/tokens)")
    parser.add_argument("--v2_only", action="store_true",
                        help="Yalnızca resmi v2 SFT asistan ağırlıklarını yükle")
    parser.add_argument("--v1_only", action="store_true",
                        help="Yalnızca v1 Base model ağırlıklarını yükle")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("❌ 'huggingface_hub' kütüphanesi bulunamadı! Yüklemek için:")
        print("   pip install huggingface_hub")
        sys.exit(1)

    print("=" * 80)
    print("🤗 Cores-AI TR — Hugging Face Model Hub Dağıtım Aracı")
    print("=" * 80)
    print(f"Hedef Depo: https://huggingface.co/{args.repo}")

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        try:
            login(token=token)
            print("🔑 Hugging Face token doğrulaması başarılı.")
        except Exception as e:
            print(f"⚠️ Token giriş uyarısı: {e}")
    else:
        print("ℹ️ Token argümanı verilmedi; sistemde kayıtlı token kontrol ediliyor...")

    api = HfApi()

    # Deponun varlığını kontrol et / oluştur
    try:
        api.repo_info(repo_id=args.repo, repo_type="model", token=token)
        print("✅ Hedef depo bulundu ve erişilebilir.")
    except Exception as e:
        print(f"ℹ️ Depo henüz mevcut değil veya erişilemedi, oluşturulmaya çalışılıyor...")
        try:
            api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, token=token)
            print("✅ Yeni model deposu başarıyla oluşturuldu!")
        except Exception as create_err:
            print(f"\n❌ Depo oluşturulamadı veya erişim reddedildi:")
            print(f"   Hata: {create_err}")
            print("\n💡 Çözüm:")
            print("   1. https://huggingface.co/settings/tokens adresinden 'Write' yetkili bir Access Token alın.")
            print(f"   2. Komutu şu şekilde çalıştırın: python tools/upload_to_hf.py --token <HF_TOKENINIZ> --repo {args.repo}")
            sys.exit(1)

    # Yüklenecek dosyaların listesi
    files_to_upload = []

    # Tokenizer (Her zaman yüklenir)
    if os.path.exists("tokenizer_tr_v2.json"):
        files_to_upload.append(("tokenizer_tr_v2.json", "tokenizer_tr_v2.json", "50.000 Kelimelik Özel Türkçe Tokenizer"))

    # v2 SFT Modelleri
    if not args.v1_only:
        sft_ckpt = "out-cores-85m-sft-v2/ckpt.pt"
        sft_best = "out-cores-85m-sft-v2/ckpt_best.pt"
        if os.path.exists(sft_ckpt):
            files_to_upload.append((sft_ckpt, "out-cores-85m-sft-v2/ckpt.pt", "v2 SFT Model Checkpoint (~978 MB)"))
        if os.path.exists(sft_best):
            files_to_upload.append((sft_best, "out-cores-85m-sft-v2/ckpt_best.pt", "v2 SFT Best Checkpoint (~978 MB)"))

    # v1 Base Model
    if not args.v2_only:
        v1_ckpt = "out-cores-85m-test/ckpt.pt"
        if os.path.exists(v1_ckpt):
            files_to_upload.append((v1_ckpt, "out-cores-85m-test/ckpt.pt", "v1 Base Model Checkpoint (~978 MB)"))

    print(f"\n📦 Toplam {len(files_to_upload)} dosya yüklenecek:")
    for local_path, repo_path, desc in files_to_upload:
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  • {local_path} ({size_mb:.1f} MB) ➔ {repo_path} [{desc}]")

    print("\n🚀 Yükleme başlatılıyor (Büyük dosyalar birkaç dakika sürebilir, lütfen bekleyin)...")
    success_count = 0
    for local_path, repo_path, desc in files_to_upload:
        try:
            print(f"\n⏳ Yükleniyor: {repo_path} ...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=args.repo,
                repo_type="model",
                token=token,
            )
            print(f"   ✅ Başarıyla yüklendi: {repo_path}")
            success_count += 1
        except Exception as up_err:
            print(f"   ❌ Yükleme başarısız ({repo_path}): {up_err}")

    print("\n" + "=" * 80)
    print(f"🎉 Dağıtım Tamamlandı! ({success_count}/{len(files_to_upload)} dosya yüklendi)")
    print(f"🌐 Modelinizi inceleyin: https://huggingface.co/{args.repo}")
    print("=" * 80)


if __name__ == "__main__":
    main()
