import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pickle
from scipy.fft import fft

# For reproducibility
np.random.seed(42)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def generate_eeg_signal(fs, duration, attention_type, noise_level=4.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    if attention_type == "Divided":
        freqs = np.random.uniform(8, 12, size=10)  # Alpha band
    elif attention_type == "Selective":
        freqs = np.random.uniform(4, 8, size=10)  # Theta band
    elif attention_type == "Executive":
        freqs = np.random.uniform(12, 30, size=10)  # Beta band
    elif attention_type == "Sustained":
        freqs = np.random.uniform(1, 4, size=10)  # Delta band
    else:
        freqs = np.random.uniform(1, 30, size=10)  # Broad spectrum for any other state

    signal = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    # Add noise to simulate real EEG; increased noise level
    noise = np.random.normal(0, noise_level, signal.shape)
    signal += noise

    signal = butter_bandpass_filter(signal, 1, 40, fs)

    return signal


fs = 256  # Sampling frequency in Hz
duration = 5  # Duration of each sample in seconds
n_samples = 500  # Reduced number of samples per attention state

attention_states = ['Divided', 'Selective', 'Executive', 'Sustained']
eeg_data = []
labels = []

# Increased noise level to reduce accuracy
noise_level = 4.0  # Adjust this value to increase noise further

for state in attention_states:
    for _ in range(n_samples):
        signal = generate_eeg_signal(fs, duration, state, noise_level=noise_level)
        eeg_data.append(signal)
        labels.append(state)

eeg_df = pd.DataFrame(eeg_data)
eeg_df['label'] = labels
eeg_df.to_csv('synthetic_eeg_attention_data.csv', index=False)

# Load the generated data
eeg_df = pd.read_csv('synthetic_eeg_attention_data.csv')

X = eeg_df.iloc[:, :-1].values
y = eeg_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def calculate_bandpower(eeg_signal, fs=256):
    freqs = np.fft.fftfreq(len(eeg_signal), 1 / fs)
    fft_values = np.abs(fft(eeg_signal)) ** 2

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
        # Drop the last three features to reduce accuracy
        bandpowers_dropped = bandpowers[:-3]  # Drop the last three features

        feature_data.append(bandpowers_dropped)

    return np.array(feature_data)


X_train_features = extract_features(X_train_scaled)
X_test_features = extract_features(X_test_scaled)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_features, y_train)
y_pred_rf = rf_clf.predict(X_test_features)

# Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train_features, y_train)
y_pred_nb = gnb.predict(X_test_features)

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_features, y_train)
y_pred_knn = knn.predict(X_test_features)

# Evaluation
for model_name, y_pred in zip(['Random Forest', 'Naive Bayes', 'KNN'], [y_pred_rf, y_pred_nb, y_pred_knn]):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# Save the Random Forest model
with open('eeg_attention_rf_classifier.pkl', 'wb') as model_file:
    pickle.dump(rf_clf, model_file)

# Load and test the Random Forest model
with open('eeg_attention_rf_classifier.pkl', 'rb') as model_file:
    loaded_rf_clf = pickle.load(model_file)

y_pred_loaded = loaded_rf_clf.predict(X_test_features)
print(f"Accuracy of the loaded Random Forest model: {accuracy_score(y_test, y_pred_loaded) * 100:.2f}%")