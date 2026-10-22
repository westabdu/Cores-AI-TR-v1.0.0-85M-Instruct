# Cores-AI TR - 85M Test Modeli - Colab T4 (16GB VRAM)
# Faz A: Pipeline dogrulama, hizli test
# Kullanim: python train.py config/train_cores_85m_test.py

out_dir = 'out-cores-85m-test'
eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = True
wandb_log = False

# Veri
dataset = 'turkce'

# Model - 85M
n_layer = 10
n_head = 10
n_embd = 640
dropout = 0.0
bias = False

# Batch - Colab T4 (16GB) için OPTIMIZE
batch_size = 16                   # 2 -> 16 (T4 rahat kaldırır)
block_size = 512                  # 256 -> 512 (daha uzun bağlam)
gradient_accumulation_steps = 5   # 20 -> 5 (efektif batch = 16 × 5 = 80)

# Optimizer
learning_rate = 3e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 5000
min_lr = 3e-5

# Sistem - Colab T4
device = 'cuda'
dtype = 'float16'                 # T4 bfloat16 desteklemez
compile = True                    # Colab'da Triton var, ÇALIŞIR! (hız)

# Gradient checkpointing - T4'te VRAM rahat, KAPAT (hız)
use_gradient_checkpointing = False