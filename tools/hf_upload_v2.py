"""
Cores-AI TR Hugging Face Model Hub Yukleyici
Python 3.15 uyumlu, pyyaml gerektirmez.
Kullanim: python tools/hf_upload_v2.py --token HF_TOKENINIZ
"""
import os, sys, argparse

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except: pass

def upload_file(local_path, repo_path, repo_id, token):
    from huggingface_hub import HfApi
    api = HfApi()
    mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  >> Yukleniyor: {repo_path} ({mb:.1f} MB) ...")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            token=token,
            repo_type="model",
        )
        print(f"  OK: {repo_path}")
        return True
    except Exception as e:
        print(f"  HATA: {e}")
        return False

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Cores-AI TR HF Yukleyici")
    parser.add_argument("--token", required=True, help="HF Write Token")
    parser.add_argument("--repo", default="aapo33/cores-ai-tr-v1.0.0-85m")
    parser.add_argument("--v2_only", action="store_true")
    parser.add_argument("--v1_only", action="store_true")
    args = parser.parse_args()

    print("="*65)
    print(" Cores-AI TR -- Hugging Face Yukleyici")
    print(f" Hedef: https://huggingface.co/{args.repo}")
    print("="*65)

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, token=args.token)
        print(" Depo hazir.")
    except Exception as e:
        print(f" Depo bilgisi: {e}")

    files = []
    if os.path.exists("tokenizer_tr_v2.json"):
        files.append(("tokenizer_tr_v2.json", "tokenizer_tr_v2.json"))
    if not args.v1_only:
        for f in ["ckpt.pt", "ckpt_best.pt"]:
            p = f"out-cores-85m-sft-v2/{f}"
            if os.path.exists(p): files.append((p, p))
    if not args.v2_only:
        p = "out-cores-85m-test/ckpt.pt"
        if os.path.exists(p): files.append((p, p))

    print(f"\n Yuklenecek {len(files)} dosya:\n")
    for l, r in files:
        print(f"  {l} -> {r}")

    ok = sum(upload_file(l, r, args.repo, args.token) for l, r in files)
    print(f"\n Tamamlandi: {ok}/{len(files)}")
    print(f" https://huggingface.co/{args.repo}")

if __name__ == "__main__":
    main()
