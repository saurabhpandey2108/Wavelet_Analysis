from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def r_squared(y_true, y_pred):
    """Custom metric for R-squared."""
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - residual / (total + tf.keras.backend.epsilon())
    return r2

def build_cnn_model(image_shape=(224, 224, 2), use_temperature_scalar=False):
    """Build a CNN for battery SOC prediction from scalogram images.

    Architecture (per paper):
        3 × (Conv2D → BatchNormalization → ReLU → MaxPooling2D)
        → Flatten → Dropout → Dense (FC) → Dense (regression output)

    Optimizer: SGDM  |  Recommended: batch_size=64, epochs=30

    Returns a compiled Keras model.
    """
    img_input = keras.Input(shape=image_shape, name='image_input')

    # --- Block 1: Conv → BN → ReLU → MaxPool ---
    x = layers.Conv2D(32, 3, padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # --- Block 2: Conv → BN → ReLU → MaxPool ---
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # --- Block 3: Conv → BN → ReLU → MaxPool ---
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # --- Flatten + Dropout + Fully Connected + Regression ---
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    if use_temperature_scalar:
        temp_input = keras.Input(shape=(1,), name='temp_input')
        x = layers.Concatenate()([x, temp_input])
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(1, activation='linear', name='soc')(x)
        model = keras.Model([img_input, temp_input], out)
    else:
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(1, activation='linear', name='soc')(x)
        model = keras.Model(img_input, out)

    # SGDM optimizer (Stochastic Gradient Descent with Momentum)
    # Reduced learning rate and added gradient clipping to prevent exploding gradients
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, clipnorm=1.0)

    # Use Adam optimizer (far more stable for this dataset size than SGDM)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse'), r_squared]
    )
    return model
