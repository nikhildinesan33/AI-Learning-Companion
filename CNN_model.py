import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPool2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPool2D(2,2),

    Flatten(),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")  # text / diagram / mixed
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.save("models/cnn_classifier.h5")

print("CNN model saved")
