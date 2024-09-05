from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import io
import base64
import pickle

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('eeg_attention_classifier.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)


def calculate_bandpower(eeg_signal, fs=256):
    freqs = np.fft.fftfreq(len(eeg_signal), 1 / fs)
    fft_values = np.abs(np.fft.fft(eeg_signal)) ** 2

    delta_band = (1, 4)
    theta_band = (4, 8)
    alpha_band = (8, 12)
    beta_band = (12, 30)

    def bandpower(freq_range):
        idx_band = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
        return np.sum(fft_values[idx_band])

    return [
        bandpower(delta_band),
        bandpower(theta_band),
        bandpower(alpha_band),
        bandpower(beta_band)
    ]


def extract_features(eeg_data, fs=256):
    feature_data = []
    for sample in eeg_data:
        bandpowers = calculate_bandpower(sample, fs)
        feature_data.append(bandpowers)
    return np.array(feature_data)


@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        X = df.iloc[:, :-1].values
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_features = extract_features(X_train)
        X_test_features = extract_features(X_test)

        y_pred = clf.predict(X_test_features)

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # Generate confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        cm_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm_image
        })
    else:
        return jsonify({'error': 'Invalid file format'})


if __name__ == '__main__':
    app.run(debug=True)