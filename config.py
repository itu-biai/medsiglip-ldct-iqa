from typing import Any, Dict

# =========================================
# Prompts
# =========================================
# Default prompt (~25 BPE tokens; the 25-35 token range had the best mean Overall
# in the prompt-length study). MedSigLIP's text encoder has a hard 64-token limit.
DEFAULT_PROMPT = (
    "Assess LDCT diagnostic image quality using noise, artifacts, contrast, "
    "sharpness, and anatomical visibility; assign MOS from 0 to 4."
)

# Detailed-rubric prompt (the "Detailed rubric" variant in the prompt study).
RUBRIC_PROMPT = (
    "Rate this low-dose CT (MOS 0-4): 0 Nondiagnostic—desired features not shown; "
    "1 Poor—diagnostic interpretation impossible; 2 Fair—limited interpretation; "
    "3 Good—diagnostic; 4 Excellent—anatomy highly visible. Return only one number 0-4."
)

# =========================================
# CONFIG
# =========================================
CONFIG: Dict[str, Any] = {
    # File paths - update these for your environment (or override via CLI flags)
    "json_path": "./data/train/train.json",
    "data_dir": "./data/train/image",
    "test_json_path": "./data/test/test.json",
    "test_data_dir": "./data/test/images",
    "checkpoint_dir": "./checkpoints",
    "exp_name": "medsiglip_iqa",  # checkpoints are saved as {exp_name}_fold{k}.pth
    # training
    "learning_rate": 1e-5,
    "batch_size": 4,
    "num_epochs": 22,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "seed": 42,
    "num_folds": 5,
    "early_stop_patience": None,  # None -> disabled (all epochs run)
    "select_metric": "val_loss",  # best-fold selection: "val_loss" or "mae"
    # image
    "image_size": 448,
    # lr scheduler
    "T_max": 100,
    "eta_min": 1e-7,
    # model
    "model_name": "google/medsiglip-448",
    "freeze_backbone": False,  # False -> vision encoder is fine-tuned end-to-end
    # AMP and grad accumulation
    "use_amp": True,
    "grad_accum_steps": 2,
    # prompt conditioning
    "use_text_prompt": True,  # Enable text prompt conditioning via FiLM
    "text_freeze": True,  # freeze text tower
    "film_strength": 1.0,  # 0 -> off
    "text_prompt": DEFAULT_PROMPT,
    "sigmoid_tau": 2.0,  # Temperature for sigmoid scaling (controls output range sharpness)
    "bound_output_0_4": True,  # Constrain output to [0, 4] range
    # loss (pure pairwise ranking by default)
    "rank_weight": 1.0,
    "mse_weight": 0.0,
    "rank_tau": 0.5,
}
