from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def build_cnn_model(image_shape=(224, 224, 2), use_temperature_scalar=False,
                    noise_std=0.0):
    """CWT-CNN regression model.

    Architecture (per paper):
        [GaussianNoise (train-only)] → 3 × (Conv2D → BN → ReLU → MaxPool)
        → Flatten → Dropout(0.5) → [concat(temp_scalar)] → Dense(64) → Dense(1)

    Parameters
    ----------
    image_shape : tuple
        (H, W, C) of the scalogram image input.
    use_temperature_scalar : bool
        If True, a second scalar input (normalized temperature) is concatenated
        with the flattened image features before the FC head.
    noise_std : float
        Standard deviation of the GaussianNoise augmentation layer applied to
        the image input. Only active during training. Set 0.0 to disable.
    """
    img_input = keras.Input(shape=image_shape, name='image_input')

    x = img_input
    if noise_std > 0:
        x = layers.GaussianNoise(noise_std, name='input_noise')(x)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    if use_temperature_scalar:
        temp_input = keras.Input(shape=(1,), name='temp_input')
        x = layers.Concatenate()([x, temp_input])
        x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        out = layers.Dense(1, activation='linear', name='soc')(x)
        model = keras.Model([img_input, temp_input], out)
    else:
        x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        out = layers.Dense(1, activation='linear', name='soc')(x)
        model = keras.Model(img_input, out)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
    )
    return model
