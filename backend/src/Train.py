import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle  # For saving the tokenizer

# Load the datasets
true_news = pd.read_csv(r"K:\fake news detection\data\True.csv")  # Update with the correct path
fake_news = pd.read_csv(r"K:\fake news detection\data\Fake.csv")    # Update with the correct path

# Add labels: 1 for true news, 0 for fake news
true_news['label'] = 1
fake_news['label'] = 0

# Combine the datasets
final_data = pd.concat([true_news, fake_news], ignore_index=True)

# Shuffle the dataset
final_data = final_data.sample(frac=1).reset_index(drop=True)

# Preprocess data
final_data['text'] = final_data['title'] + " " + final_data['text']
X = final_data['text']
y = final_data['label']

# Check for NaN values
if X.isnull().any() or y.isnull().any():
    print("Found NaN values in the dataset.")
else:
    print("No NaN values found in the dataset.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and padding
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build the model
model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)))  # Using Bidirectional LSTM
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_pad,
    y_train,
    epochs=10,  # Adjust the number of epochs as necessary
    batch_size=64,
    validation_data=(X_test_pad, y_test),
    verbose=2  # Verbosity level
)

# Save the model and tokenizer
model.save(r"K:\fake news detection\model\fake_news_detection_model_BILSTM.h5")
with open(r"K:\fake news detection\tokenizer.pickle", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Plotting the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
