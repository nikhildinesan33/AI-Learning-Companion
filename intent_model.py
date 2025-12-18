import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
os.makedirs("models", exist_ok=True)

texts = [
    "explain this",
    "summarize this",
    "give me a quiz",
    "explain for beginners",
    "explain in detail"
]

labels = [0, 1, 2, 3, 4]

intent_map = {
    0: "Explain",
    1: "Summary",
    2: "Quiz",
    3: "Beginner",
    4: "Advanced"
}

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=5)
y = np.array(labels)

model = Sequential()
model.add(Embedding(1000, 64, input_length=5))
model.add(LSTM(32))
model.add(Dense(5, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, epochs=50, verbose=0)
model.save("models/intent_lstm.h5")

print("Intent model trained & saved")
