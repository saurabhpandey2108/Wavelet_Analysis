from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def build_cnn_model(image_shape=(224, 224, 2), use_temperature_scalar=False):
    """Build a CNN for battery SOC prediction from scalogram images.

    Architecture (per paper):
        3 × (Conv2D → BatchNormalization → ReLU → MaxPooling2D)
        → Flatten → Dropout → Dense (FC) → Dense (regression output)

    Returns a compiled Keras model.
    """
    img_input = keras.Input(shape=image_shape, name='image_input')

    # --- Block 1: Conv → BN → ReLU → MaxPool ---
    x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # --- Block 2: Conv → BN → ReLU → MaxPool ---
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # --- Block 3: Conv → BN → ReLU → MaxPool ---
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # --- Flatten + Dropout + Fully Connected + Regression ---
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

    # Use Adam optimizer with lower learning rate and gradient clipping to avoid model collapse
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model
