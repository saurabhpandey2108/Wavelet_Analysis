Project CWT helpers

Folders created:
- `data_preprocessing/`  : functions to load CSV, estimate fs, and window signals
- `cwt_calc/`           : compute_cwt helper (uses Morlet by default)
- `cwt_image/`          : convert CWT coeffs to image/scalogram, with optional temperature channel
- `cnn_model/`          : Keras model skeleton that can take image and optional temperature scalar

Quick usage example (from a notebook):

from data_preprocessing.preprocess import load_signals, create_windows, estimate_fs
from cwt_image.image_utils import cwt_to_image, stack_channels
from cnn_model.model import build_cnn_model

# load
_, voltage, current, temperature, time = load_signals('dataset/00005.csv')
fs = estimate_fs(time)

# window
v_w = create_windows(voltage, 256, 128)
i_w = create_windows(current, 256, 128)
t_w = create_windows(temperature, 256, 128)

# cwt -> image
img_v = cwt_to_image(v_w[0], fs)  # returns HxW
img_i = cwt_to_image(i_w[0], fs)
# combine with temperature as extra channel
cwt_img = stack_channels(img_v, img_i, temperature_value=float(t_w[0].mean()))

# build model
model = build_cnn_model(image_shape=cwt_img.shape, use_temperature_scalar=False)

Notes:
- The helpers use Morlet ('morl') wavelet by default.
- The image helpers provide an option to append a normalized temperature channel.
- If you meant something else by "use temperature as scale" please clarify —
  do you want temperature to modify the wavelet scales, or be used as an input channel/scalar to the CNN?
