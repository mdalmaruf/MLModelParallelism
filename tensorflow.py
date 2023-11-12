import tensorflow as tf

# Define a simple multi-layer model
# Place each layer on a different GPU manually

# Assuming you have two GPUs available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to true
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device('/GPU:0'):
    layer1 = tf.keras.layers.Dense(128, activation='relu')

with tf.device('/GPU:1'):
    layer2 = tf.keras.layers.Dense(64, activation='relu')

with tf.device('/GPU:0'):
    output_layer = tf.keras.layers.Dense(10, activation='softmax')

# Create a model using the layers above
def create_model():
    inputs = tf.keras.Input(shape=(784,))
    x = layer1(inputs)
    x = layer2(x)
    outputs = output_layer(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Compile and train the model
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dummy dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255

# Train the model
model.fit(x_train, y_train, epochs=5)
