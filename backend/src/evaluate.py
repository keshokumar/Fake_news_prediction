import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model_path = r"K:\fake news detection\backend\model\fake_news_detection_model_BILSTM.h5"
model = tf.keras.models.load_model(model_path)

# Load the tokenizer
with open(r"K:\fake news detection\tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load test data
fake_data = pd.read_csv(r"K:\fake news detection\backend\data\Fake.csv")
real_data = pd.read_csv(r"K:\fake news detection\backend\data\True.csv")

# Label the data
fake_data['label'] = 0  # Fake news label
real_data['label'] = 1   # True news label

# Combine the datasets and shuffle
final_data = pd.concat([fake_data, real_data]).sample(frac=1).reset_index(drop=True)

# Prepare the text data
final_data['text'] = final_data['title'] + " " + final_data['text']
X = final_data['text']
y = final_data['label']

# Tokenization and padding (same as in training)
max_len = 200
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len)

# Make predictions
y_pred_proba = model.predict(X_pad)
y_pred = (y_pred_proba > 0.5).astype(int)

# Classification report
print("Classification Report:")
print(classification_report(y, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions')
plt.ylabel('Actuals')
plt.title('Confusion Matrix')
plt.show()

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC
print(f"AUC: {roc_auc:.2f}")
