import tensorflow as tf

def build_unet(input_shape=(256, 256, 3)):
    inputs = tf.keras.layers.Input(input_shape)
    # Example U-Net structure
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    # Add downsampling, bottleneck, and upsampling layers here
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c1)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])
