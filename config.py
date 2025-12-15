from typing import Dict, Any

# =========================================
# CONFIG
# =========================================
CONFIG: Dict[str, Any] = {
    # File paths - Update these paths according to your environment
    "checkpoint_path": "./checkpoints/medsiglip_iqa",
    "json_path": "./data/train.json",
    "data_dir": "./data/train/images",
    "test_json_path": "./data/test.json",
    "test_data_dir": "./data/test/images",
    # training
    "learning_rate": 1e-5,
    "batch_size": 4,
    "num_epochs": 22,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "seed": 42,
    # image
    "image_size": 448,
    # lr scheduler
    "T_max": 100,
    "eta_min": 1e-7,
    # model
    "freeze_backbone": False,
    "medsiglip_model": "google/medsiglip-448",
    "model_name": "google/medsiglip-448",
    # AMP and grad accumulation
    "use_amp": True,
    "grad_accum_steps": 2,
    # prompt conditioning
    "use_text_prompt": True,  # Enable text prompt conditioning via FiLM
    "text_freeze": True,  # freeze text tower
    "film_strength": 1.0,  # 0 -> off
    "text_prompt": "Rate this low-dose CT (MOS 0-4): 0 Nondiagnostic—desired features not shown; 1 Poor—diagnostic interpretation impossible; 2 Fair—limited interpretation; 3 Good—diagnostic; 4 Excellent—anatomy highly visible. Return only one number 0-4.",
    "sigmoid_tau": 2.0,  # Temperature for sigmoid scaling (controls output range sharpness)
    "bound_output_0_4": True,  # Constrain output to [0, 4] range
}
