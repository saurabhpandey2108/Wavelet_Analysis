import os
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Global Pipeline Constants
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
INITIAL_SOC = 0.8          # All datasets start at 80% SOC
CAPACITY_AH = 2.0          # SP20 rated capacity (Ah)
WINDOW_SIZE = 256          # Samples per window
STRIDE = 5                 # Heavy overlap to artificially increase dataset size (~2000 windows)
SCALES = np.arange(1, 128) # CWT scales
EPOCHS = 30
BATCH_SIZE = 64            # Paper: minibatch size of 64
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
#  Dataset Configurations (per paper methodology)
# ──────────────────────────────────────────────────────────────────────────────

# TRAINING: DST at 0°C ambient
TRAIN_DATA = {
    "path": os.path.join("dataset", "DST", "02_24_2016_SP20-2_0C_DST_80SOC.xls"),
    "sheet": "Channel_1-006",
    "ambient_temp": 0.0,
    "label": "DST @ 0°C",
}

# TESTING: DST, US06, FUDS at 25°C ambient
TEST_DATA = [
    {
        "path": os.path.join("dataset", "DST", "11_05_2015_SP20-2_DST_80SOC.xls"),
        "sheet": "Channel_1-008",
        "ambient_temp": 25.0,
        "label": "DST @ 25°C",
    },
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
