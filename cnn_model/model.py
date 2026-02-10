from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(image_shape=(224, 224, 2), use_temperature_scalar=False):
    """Build a small CNN that can accept image input and optional scalar temperature input.

    Returns a compiled Keras model.
    """
    img_input = keras.Input(shape=image_shape, name='image_input')

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(img_input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

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

    model.compile(optimizer='adam', loss='mse')
    return model
