import os
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Global Pipeline Constants
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
INITIAL_SOC = 0.8           # Dynamic profile (Step 7) starts at 80% SOC
CAPACITY_AH = 2.0           # SP20 rated capacity (Ah)
WINDOW_SIZE = 1280           # ≈1300 s at fs≈0.985 Hz → ~3.5 DST cycles (period 360 s)
STRIDE = 256                 # 80% overlap; yields ~33 windows per DST Step-7 record
SCALES = np.geomspace(2.0, 300.0, 96)  # log-spaced; max=300 → pseudo-freq ~0.003 Hz
                                         # (just below 1/360 s DST fundamental).
                                         # COI-safe since a_max < n/(2√2) ≈ 452.
EPOCHS = 30
BATCH_SIZE = 64
RANDOM_SEED = 42
STEP_FILTER = 7             # Arbin Step_Index for the dynamic driving profile

# Training-side regularization
USE_TEMPERATURE = True      # Meaningful now — training mixes 0°C and 25°C
NOISE_STD = 0.05            # GaussianNoise std on normalized [0,1] images
EARLY_STOP_PATIENCE = 8     # val_loss plateau patience before early stop
TEMP_NORM_DENOM = 25.0      # 0°C → 0.0, 25°C → 1.0 on the temperature input

# ──────────────────────────────────────────────────────────────────────────────
#  Dataset Configurations
# ──────────────────────────────────────────────────────────────────────────────
#  Training uses both DST cycles (0°C and 25°C) to expose the model to
#  cross-temperature scalogram variation. Testing uses US06 and FUDS @ 25°C
#  to measure generalization to *new dynamic profiles* at a trained temperature.

TRAIN_DATASETS = [
    {
        "path": os.path.join("dataset", "DST", "02_24_2016_SP20-2_0C_DST_80SOC.xls"),
        "sheet": "Channel_1-006",
        "ambient_temp": 0.0,
        "label": "DST @ 0°C",
    },
    {
        "path": os.path.join("dataset", "DST", "11_05_2015_SP20-2_DST_80SOC.xls"),
        "sheet": "Channel_1-008",
        "ambient_temp": 25.0,
        "label": "DST @ 25°C",
    },
]

TEST_DATASETS = [
    {
        "path": os.path.join("dataset", "US06", "11_11_2015_SP20-2_US06_80SOC.xls"),
        "sheet": "Channel_1-008",
        "ambient_temp": 25.0,
        "label": "US06 @ 25°C",
    },
    {
        "path": os.path.join("dataset", "FUDS", "11_06_2015_SP20-2_FUDS_80SOC.xls"),
        "sheet": "Channel_1-008",
        "ambient_temp": 25.0,
        "label": "FUDS @ 25°C",
    },
]
