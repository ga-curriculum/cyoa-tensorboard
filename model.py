import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = (X_train / 255).reshape(-1, 28, 28, 1)
X_test = (X_test / 255).reshape(-1, 28, 28, 1)

# Define the neural network model
model = Sequential()
model.add(Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  metrics=['acc'])

# Create a TensorBoard callback
log_dir = "logs/fit1"  # Directory where the logs will be stored
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=512,
    epochs=30,
    callbacks=[tb]
)

# To run TensorBoard, open a terminal and navigate to your project directory.
# Then, run the following command:
# tensorboard --logdir=logs/fit

# You can access TensorBoard in your web browser at http://localhost:6006/
