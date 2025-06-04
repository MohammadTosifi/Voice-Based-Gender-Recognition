import os
import random
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
import joblib
import argparse # Added for command-line arguments
import warnings
import librosa.effects
import librosa.filters
from scipy.signal import butter, filtfilt
import noisereduce as nr



# Constants
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 5  # seconds
SAMPLE_SIZE = 5100    # samples per gender




# Add these constants after your other constants
FEATURE_GROUPS = {
    'MFCC': (0, 13),
    'Mel': (13, 141),
    'Chroma': (141, 153),
    'Contrast': (153, 160),
    'Tonnetz': (160, 166)
}

def analyze_feature_correlations(features, labels):
    # 1. Scale features BEFORE analysis
    # 1. Scale features BEFORE analysis
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 2. Use mean of features within each group as representative
    group_data = {}
    for group_name, (start, end) in FEATURE_GROUPS.items():
        # Calculate the mean of all features in the current group for each sample
        group_data[group_name] = np.mean(features_scaled[:, start:end], axis=1)  
    
    # 3. Improved correlation calculation
    groups = list(FEATURE_GROUPS.keys())
    r2_matrix = np.zeros((len(groups), len(groups)))
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i != j:
                # Use representative features with Pearson correlation
                r_val = np.corrcoef(group_data[group1], group_data[group2])[0,1]
                r2_matrix[i,j] = r_val**2
    
    # Visualization (unchanged)
    plt.figure(figsize=(10,8))
    sns.heatmap(r2_matrix, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=groups, yticklabels=groups)
    plt.title("Feature Group Correlations (R²)")
    plt.show()
    
    # Print key correlations
    print("\nKey Correlations (Fixed):")
    print(f"MFCC-Chroma: {r2_matrix[groups.index('MFCC'), groups.index('Chroma')]:.4f}")
    print(f"Chroma-Contrast: {r2_matrix[groups.index('Chroma'), groups.index('Contrast')]:.4f}")
    print(f"MFCC-Tonnetz: {r2_matrix[groups.index('MFCC'), groups.index('Tonnetz')]:.4f}")
    print(f"Chroma-Mel: {r2_matrix[groups.index('Chroma'), groups.index('Mel')]:.4f}")



# 1. Remove duplicates and load balanced dataset
def load_balanced_dataset(base_dir="data"):
    female_dir = os.path.join(base_dir, "female")
    male_dir = os.path.join(base_dir, "male")
    
    def get_unique_files(directory, max_files):
        unique_files = []
        unique_hashes = set()
        
        for file in os.listdir(directory):
            if len(unique_files) >= max_files:
                break
                
            if file.endswith(".wav"):
                file_path = os.path.join(directory, file)
                try:
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash not in unique_hashes:
                        unique_hashes.add(file_hash)
                        unique_files.append(file_path)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
        
        return unique_files
    
    # Get unique files (1000 per gender)
    female_files = get_unique_files(female_dir, SAMPLE_SIZE)
    male_files = get_unique_files(male_dir, SAMPLE_SIZE)
    
    # Combine and create labels (0=female, 1=male)
    all_files = female_files + male_files
    labels = [0] * len(female_files) + [1] * len(male_files)
    
    return all_files, labels

# 2. Feature Extraction (MFCC, Mel, Chroma, Contrast, Tonnetz)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)
        
        target_samples_per_chunk = SAMPLE_RATE * MAX_AUDIO_LENGTH
        
        all_feature_chunks = []
        
        if len(y) == 0:
            print(f"Warning: Empty audio data for {file_path}, skipping feature extraction.")
            return all_feature_chunks

        num_total_samples = len(y)
        num_chunks = int(np.ceil(num_total_samples / target_samples_per_chunk))
        
        if num_chunks == 0 and num_total_samples > 0: # Should ideally be at least 1 if there's data
            num_chunks = 1
        elif num_chunks == 0 and num_total_samples == 0: # No data, no chunks
             return all_feature_chunks


        for i in range(num_chunks):
            start_sample = i * target_samples_per_chunk
            end_sample = start_sample + target_samples_per_chunk
            segment = y[start_sample:end_sample]
            
            # Pad if segment is shorter than target_samples_per_chunk (common for the last chunk or short audio)
            if len(segment) < target_samples_per_chunk:
                padding_needed = target_samples_per_chunk - len(segment)
                segment = np.pad(segment, (0, padding_needed), mode='constant')
            elif len(segment) > target_samples_per_chunk: # Should not happen with ceil logic but safeguard
                segment = segment[:target_samples_per_chunk]

            # MFCC with delta features
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # Mel-spectrogram
            mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel)
            
            # Chroma, Spectral Contrast, Tonnetz
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
            
            # Aggregate features by taking mean over time for the current chunk
            current_chunk_features = np.concatenate([
                np.mean(mfcc, axis=1),
                np.mean(mfcc_delta, axis=1),
                np.mean(mel_db, axis=1),
                np.mean(chroma, axis=1),
                np.mean(contrast, axis=1),
                np.mean(tonnetz, axis=1)
            ])
            all_feature_chunks.append(current_chunk_features)
            
        return all_feature_chunks
        
    except Exception as e:
        print(f"Error processing {file_path} during feature extraction: {e}")
        return [] # Return empty list on error

# 3. Dataset Class for PyTorch
class VoiceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])

# 4. CNN Model
class VoiceGenderCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Layer 1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3) # Added dropout

        # Layer 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3) # Added dropout
        
        # Calculate the linear layer input size
        # Input shape to conv1: (batch, 1, input_dim)
        # After conv1: (batch, 32, input_dim)
        # After bn1: (batch, 32, input_dim)
        # After pool1: (batch, 32, input_dim // 2)
        # After dropout1: (batch, 32, input_dim // 2)
        # After conv2: (batch, 64, input_dim // 2)
        # After bn2: (batch, 64, input_dim // 2)
        # After pool2: (batch, 64, (input_dim // 2) // 2) = (batch, 64, input_dim // 4)
        # After dropout2: (batch, 64, input_dim // 4)
        conv_output_size = 64 * (input_dim // 4)
        
        self.fc1 = nn.Linear(conv_output_size, 128) # Increased size of fc1
        self.dropout_fc = nn.Dropout(0.5) # Dropout before final layer
        self.fc2 = nn.Linear(128, 2)  # 2 classes (male/female)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, features) -> (batch, 1, features)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# Function to predict on new audio files
def predict_on_audio_folder(audio_dir, scaler, svm_model, rf_ensemble, cnn_model, device):
    print(f"\nPredicting on new audio files from '{audio_dir}' (using chunking for long files)...")
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    if not audio_files:
        print(f"No .wav files found in '{audio_dir}'.")
        return

    target_names = ['female', 'male']
    # Ensure cnn_model is in eval mode
    cnn_model.eval()

    all_true_labels = []
    all_svm_preds = []
    all_rf_preds = []
    all_cnn_preds = []

    for audio_file in audio_files:
        file_path = os.path.join(audio_dir, audio_file)
        file_name_lower = os.path.basename(file_path).lower()
        
        true_label = -1 # Default for uninferrable
        if "female" in file_name_lower:
            true_label = 0
        elif "male" in file_name_lower or "alex" in file_name_lower:
            true_label = 1
        
        if true_label == -1:
            print(f"Could not infer label for {os.path.basename(file_path)}, skipping.")
            continue
            
        # Use enhanced feature extraction
        chunk_feature_arrays = extract_enhanced_features(file_path)

        if not chunk_feature_arrays:
            print(f"Could not extract features for {os.path.basename(file_path)}, skipping.")
            continue

        svm_chunk_preds = []
        rf_chunk_preds = []
        cnn_chunk_preds = []

        for single_chunk_features in chunk_feature_arrays:
            # Scale (reshape for scaler as it expects 2D array)
            single_chunk_features_scaled = scaler.transform(single_chunk_features.reshape(1, -1))

            # SVM Prediction for the chunk
            svm_pred = svm_model.predict(single_chunk_features_scaled)[0]
            svm_chunk_preds.append(svm_pred)

            # Update RF prediction to use ensemble
            ensemble_preds = []
            for i, model in enumerate(rf_ensemble['models']):
                if rf_ensemble['feature_indices'][i] is None:
                    pred = model.predict(single_chunk_features_scaled)[0]
                else:
                    pred = model.predict(single_chunk_features_scaled[:, rf_ensemble['feature_indices'][i]])[0]
                ensemble_preds.append(pred)
            
            # Weighted voting
            weighted_pred = np.sum([p * w for p, w in zip(ensemble_preds, rf_ensemble['weights'])])
            rf_chunk_preds.append(int(weighted_pred > 0.5))

            # CNN Prediction for the chunk
            cnn_features_tensor = torch.FloatTensor(single_chunk_features_scaled).to(device)
            with torch.no_grad():
                outputs = cnn_model(cnn_features_tensor) # CNN's forward handles unsqueeze
                _, cnn_pred_tensor = torch.max(outputs, 1)
                cnn_pred = cnn_pred_tensor.cpu().numpy()[0]
                cnn_chunk_preds.append(cnn_pred)
        
        # Aggregate predictions for the file using mode
        if not svm_chunk_preds: # Should only happen if chunk_feature_arrays was empty and already handled
            print(f"No chunk predictions available for {os.path.basename(file_path)}.")
            continue
            
        # scipy.stats.mode returns ModeResult object.
        # Use np.atleast_1d to handle cases where .mode might be scalar.
        final_svm_pred = int(np.atleast_1d(stats.mode(svm_chunk_preds, keepdims=False).mode)[0])
        final_rf_pred = int(np.atleast_1d(stats.mode(rf_chunk_preds, keepdims=False).mode)[0])
        final_cnn_pred = int(np.atleast_1d(stats.mode(cnn_chunk_preds, keepdims=False).mode)[0])
        
        all_true_labels.append(true_label)
        all_svm_preds.append(final_svm_pred)
        all_rf_preds.append(final_rf_pred)
        all_cnn_preds.append(final_cnn_pred)

        file_name_display = os.path.basename(file_path)
        svm_label_display = target_names[final_svm_pred]
        rf_label_display = target_names[final_rf_pred]
        cnn_label_display = target_names[final_cnn_pred]
        true_label_display = target_names[true_label]
        num_processed_chunks = len(chunk_feature_arrays)
        
        print(f"File: {file_name_display} ({num_processed_chunks} chunk(s)) -> True: {true_label_display}, SVM: {svm_label_display}, RF: {rf_label_display}, CNN: {cnn_label_display}")

    if not all_true_labels:
        print("No files in 'audio' folder could be processed or have inferable labels.")
        return

    print("\n--- Performance on 'audio' directory ---")
    print(f"Processed {len(all_true_labels)} files from '{audio_dir}'.")

    print("\nSVM Results on 'audio' directory:")
    print(f"Accuracy: {accuracy_score(all_true_labels, all_svm_preds):.4f}")
    print(classification_report(all_true_labels, all_svm_preds, target_names=target_names, zero_division=0))

    print("\nRandom Forest Results on 'audio' directory:")
    print(f"Accuracy: {accuracy_score(all_true_labels, all_rf_preds):.4f}")
    print(classification_report(all_true_labels, all_rf_preds, target_names=target_names, zero_division=0))

    print("\nCNN Results on 'audio' directory:")
    print(f"Accuracy: {accuracy_score(all_true_labels, all_cnn_preds):.4f}")
    print(classification_report(all_true_labels, all_cnn_preds, target_names=target_names, zero_division=0))

# Main Execution

# Define these at the module level so they are accessible by the functions
MODEL_SAVE_DIR = "saved_models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MFCC (13) + Mel (128) + Chroma (12) + Contrast (7) + Tonnetz (6) = 166
CNN_INPUT_DIM = 166

def perform_training_and_evaluation(audio_dir_for_post_training_eval="audio"):
    """
    Handles the complete model training pipeline, including data loading, 
    feature extraction, model training (SVM, RF, CNN), saving models,
    and evaluating on a test split and an external audio directory.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print("Starting model training and evaluation pipeline...")

    # 1. Load and balance dataset
    print("Loading and balancing dataset...")
    files, labels = load_balanced_dataset()
    print(f"Total samples: {len(files)} ({labels.count(0)} female, {labels.count(1)} male)")
    
    # 2. Extract features
    print("Extracting features...")
    features_list_for_training = [] 
    valid_files = []
    valid_labels = []
    
    for file, label in zip(files, labels):
        # Use the new enhanced feature extraction
        extracted_chunks = extract_enhanced_features(file) 
        
        if extracted_chunks:
            first_chunk_features = extracted_chunks[0]
            features_list_for_training.append(first_chunk_features)
            valid_files.append(file)
            valid_labels.append(label)
            
    if not features_list_for_training:
        print("Error: No features could be extracted for training. Exiting.")
        return

    features = np.array(features_list_for_training)
    labels = np.array(valid_labels)
    
    if features.shape[1] != CNN_INPUT_DIM:
        print(f"Warning: Extracted feature dimension {features.shape[1]} does not match expected CNN_INPUT_DIM {CNN_INPUT_DIM}.")
        return
        
    print(f"Feature shape for training: {features.shape}")
    
    print("\nAnalyzing feature correlations...")
    analyze_feature_correlations(features, labels)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    print(f"Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.joblib'))
    print(f"Scaler saved to {os.path.join(MODEL_SAVE_DIR, 'scaler.joblib')}")
    
    # 5. Train traditional ML models
    print("\nTraining SVM...")
    svm_model = SVC(kernel='rbf', C=1.0, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    joblib.dump(svm_model, os.path.join(MODEL_SAVE_DIR, 'svm_model.joblib'))
    print(f"SVM model saved to {os.path.join(MODEL_SAVE_DIR, 'svm_model.joblib')}")

    svm_train_pred = svm_model.predict(X_train_scaled)
    svm_train_accuracy = accuracy_score(y_train, svm_train_pred)
    print(f"SVM Training Accuracy: {svm_train_accuracy:.4f}")

    svm_pred = svm_model.predict(X_test_scaled)
    print("SVM Results (on test split):")
    print(classification_report(y_test, svm_pred, target_names=['female', 'male']))
    
    print("\nTraining Random Forest...")
    # Create an ensemble of RF models with different feature subsets
    rf_models = []
    rf_weights = []
    
    # Model 1: Full feature set (166 features)
    rf1 = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        bootstrap=True,
        max_features='sqrt'
    )
    rf1.fit(X_train_scaled, y_train)
    rf_models.append(rf1)
    rf_weights.append(0.4) # Weight for the full model
    
    # Model 2: Focus on MFCC and Mel features
    mfcc_mel_indices = np.concatenate([
        np.arange(FEATURE_GROUPS['MFCC'][0], FEATURE_GROUPS['MFCC'][1]),
        np.arange(FEATURE_GROUPS['Mel'][0], FEATURE_GROUPS['Mel'][1])
    ])
    rf2 = RandomForestClassifier(
        n_estimators=200, # Adjusted estimators
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=43, # Different random state
        bootstrap=True,
        max_features='sqrt'
    )
    rf2.fit(X_train_scaled[:, mfcc_mel_indices], y_train)
    rf_models.append(rf2)
    rf_weights.append(0.3) # Weight for MFCC+Mel model
    
    # Model 3: Focus on Chroma, Contrast, and Tonnetz features
    chroma_contrast_tonnetz_indices = np.concatenate([
        np.arange(FEATURE_GROUPS['Chroma'][0], FEATURE_GROUPS['Chroma'][1]),
        np.arange(FEATURE_GROUPS['Contrast'][0], FEATURE_GROUPS['Contrast'][1]),
        np.arange(FEATURE_GROUPS['Tonnetz'][0], FEATURE_GROUPS['Tonnetz'][1])
    ])
    rf3 = RandomForestClassifier(
        n_estimators=250, # Adjusted estimators
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=44, # Different random state
        bootstrap=True,
        max_features='sqrt'
    )
    rf3.fit(X_train_scaled[:, chroma_contrast_tonnetz_indices], y_train)
    rf_models.append(rf3)
    rf_weights.append(0.3) # Weight for Chroma+Contrast+Tonnetz model
    
    # Save the ensemble
    rf_ensemble = {
        'models': rf_models,
        'weights': rf_weights,
        'feature_indices': [None, mfcc_mel_indices, chroma_contrast_tonnetz_indices] # Updated feature indices
    }
    joblib.dump(rf_ensemble, os.path.join(MODEL_SAVE_DIR, 'rf_ensemble.joblib'))
    print(f"Random Forest ensemble saved to {os.path.join(MODEL_SAVE_DIR, 'rf_ensemble.joblib')}")
    
    # Evaluate ensemble on training set
    rf_train_preds = []
    for i, model in enumerate(rf_models):
        if rf_ensemble['feature_indices'][i] is None:
            pred = model.predict(X_train_scaled)
        else:
            pred = model.predict(X_train_scaled[:, rf_ensemble['feature_indices'][i]])
        rf_train_preds.append(pred)
    
    # Weighted voting
    rf_train_pred = np.zeros(len(y_train))
    for i, preds in enumerate(rf_train_preds):
        rf_train_pred += rf_ensemble['weights'][i] * preds
    rf_train_pred = (rf_train_pred > 0.5).astype(int)
    
    rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
    print(f"Random Forest Training Accuracy: {rf_train_accuracy:.4f}")
    
    # Evaluate on test set
    rf_test_preds = []
    for i, model in enumerate(rf_models):
        if rf_ensemble['feature_indices'][i] is None:
            pred = model.predict(X_test_scaled)
        else:
            pred = model.predict(X_test_scaled[:, rf_ensemble['feature_indices'][i]])
        rf_test_preds.append(pred)
    
    # Weighted voting for test set
    rf_test_pred = np.zeros(len(y_test))
    for i, preds in enumerate(rf_test_preds):
        rf_test_pred += rf_ensemble['weights'][i] * preds
    rf_test_pred = (rf_test_pred > 0.5).astype(int)
    
    print("Random Forest Results (on test split):")
    print(classification_report(y_test, rf_test_pred, target_names=['female', 'male']))
    
    # 6. Train CNN
    print("\nTraining CNN...")
    train_dataset = VoiceDataset(X_train_scaled, y_train)
    test_dataset = VoiceDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    cnn_model = VoiceGenderCNN(input_dim=X_train_scaled.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0005) # Slightly reduced learning rate
    
    cnn_model.to(device)
    
    # Consider adding early stopping or adjusting epochs based on validation loss if available
    for epoch in range(25): # Slightly increased epochs
        cnn_model.train()
        running_loss = 0.0
        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/25, Loss: {running_loss/len(train_loader):.4f}")
    
    torch.save(cnn_model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'cnn_model.pth'))
    print(f"CNN model saved to {os.path.join(MODEL_SAVE_DIR, 'cnn_model.pth')}")

    cnn_model.eval() 
    cnn_train_all_preds = []
    cnn_train_all_labels = []
    # Re-create DataLoader for full training set accuracy without shuffle
    cnn_train_dataset_for_acc = VoiceDataset(X_train_scaled, y_train) 
    cnn_train_loader_for_acc = DataLoader(cnn_train_dataset_for_acc, batch_size=32, shuffle=False)

    with torch.no_grad():
        for inputs, labels_batch in cnn_train_loader_for_acc:
            inputs, labels_batch = inputs.to(device), labels_batch.squeeze().to(device)
            outputs = cnn_model(inputs)
            _, preds = torch.max(outputs, 1)
            cnn_train_all_preds.extend(preds.cpu().numpy())
            cnn_train_all_labels.extend(labels_batch.cpu().numpy())
    cnn_train_accuracy = accuracy_score(cnn_train_all_labels, cnn_train_all_preds)
    print(f"CNN Training Accuracy: {cnn_train_accuracy:.4f}")
    
    cnn_model.eval() 
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for inputs, labels_batch in test_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.squeeze().to(device) # .squeeze() for labels
            outputs = cnn_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels_batch.cpu().numpy())
    
    print("\nCNN Results (on test split):")
    print(classification_report(all_labels_test, all_preds_test, target_names=['female', 'male']))

    # 7. Predict on the specified audio folder using newly trained models
    print(f"\n--- Running prediction on '{audio_dir_for_post_training_eval}' after training ---")
    predict_on_audio_folder(audio_dir_for_post_training_eval, scaler, svm_model, rf_ensemble, cnn_model, device)
    print("Training and evaluation pipeline completed.")

def perform_prediction_with_saved_models(audio_dir_to_predict="audio"):
    """
    Loads pre-trained models and scaler, then performs predictions 
    on audio files in the specified directory.
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Mode: Predict. Loading models from '{MODEL_SAVE_DIR}' and predicting on '{audio_dir_to_predict}'.")

    # Load scaler
    scaler_path = os.path.join(MODEL_SAVE_DIR, 'scaler.joblib')
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at {scaler_path}. Please train models first.")
        return

    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")

    # Load SVM model
    svm_model_path = os.path.join(MODEL_SAVE_DIR, 'svm_model.joblib')
    if not os.path.exists(svm_model_path):
        print(f"Error: SVM model not found at {svm_model_path}. Please train models first.")
        return
    svm_model = joblib.load(svm_model_path)
    print(f"SVM model loaded from {svm_model_path}")

    # Load Random Forest model
    rf_ensemble_path = os.path.join(MODEL_SAVE_DIR, 'rf_ensemble.joblib')
    if not os.path.exists(rf_ensemble_path):
        print(f"Error: Random Forest ensemble not found at {rf_ensemble_path}. Please train models first.")
        return
    rf_ensemble = joblib.load(rf_ensemble_path)
    print(f"Random Forest ensemble loaded from {rf_ensemble_path}")

    # Load CNN model
    cnn_model_path = os.path.join(MODEL_SAVE_DIR, 'cnn_model.pth')
    if not os.path.exists(cnn_model_path):
        print(f"Error: CNN model not found at {cnn_model_path}. Please train models first.")
        return
    cnn_model = VoiceGenderCNN(input_dim=CNN_INPUT_DIM) # CNN_INPUT_DIM is now module-level
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device)) # device is module-level
    cnn_model.to(device)
    cnn_model.eval() 
    print(f"CNN model loaded from {cnn_model_path}")
    
    # Predict on the specified 'audio' folder
    predict_on_audio_folder(audio_dir_to_predict, scaler, svm_model, rf_ensemble, cnn_model, device)
    print(f"Prediction pipeline completed for directory: {audio_dir_to_predict}")

def generate_analysis_figures():
    """
    Generates and saves all key analysis figures as PNGs in a 'figures' directory.
    """
    warnings.filterwarnings('ignore')

    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)

    # Load and balance dataset
    files, labels = load_balanced_dataset()
    features_list_for_training = []
    valid_labels = []
    for file, label in zip(files, labels):
        # Use enhanced feature extraction
        extracted_chunks = extract_enhanced_features(file)
        if extracted_chunks:
            first_chunk_features = extracted_chunks[0]
            features_list_for_training.append(first_chunk_features)
            valid_labels.append(label)
    features = np.array(features_list_for_training)
    labels = np.array(valid_labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    svm_model = SVC(kernel='rbf', C=1.0, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    svm_train_acc = accuracy_score(y_train, svm_model.predict(X_train_scaled))
    svm_test_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))

    # Train RF
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train_scaled))
    rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))

    # Train CNN
    class VoiceDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])
    class VoiceGenderCNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(2)
            self.dropout1 = nn.Dropout(0.3)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(2)
            self.dropout2 = nn.Dropout(0.3)
            conv_output_size = 64 * (input_dim // 4)
            self.fc1 = nn.Linear(conv_output_size, 128)
            self.dropout_fc = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 2)
        def forward(self, x):
            x = x.unsqueeze(1)
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = self.dropout1(x)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = self.dropout2(x)
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.dropout_fc(x)
            x = self.fc2(x)
            return x
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VoiceDataset(X_train_scaled, y_train)
    test_dataset = VoiceDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    cnn_model = VoiceGenderCNN(input_dim=X_train_scaled.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0005)
    cnn_model.to(device)
    for epoch in range(10):
        cnn_model.train()
        for inputs, labels_batch in train_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.squeeze().to(device)
            optimizer.zero_grad()
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
    cnn_model.eval()
    with torch.no_grad():
        cnn_train_preds = []
        for inputs, labels_batch in train_loader:
            inputs = inputs.to(device)
            outputs = cnn_model(inputs)
            _, preds = torch.max(outputs, 1)
            cnn_train_preds.extend(preds.cpu().numpy())
        cnn_test_preds = []
        for inputs, labels_batch in test_loader:
            inputs = inputs.to(device)
            outputs = cnn_model(inputs)
            _, preds = torch.max(outputs, 1)
            cnn_test_preds.extend(preds.cpu().numpy())
    cnn_train_acc = accuracy_score(y_train, cnn_train_preds)
    cnn_test_acc = accuracy_score(y_test, cnn_test_preds)

    # 5. Feature Group Correlations
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    group_data = {}
    for group_name, (start, end) in FEATURE_GROUPS.items():
        group_data[group_name] = np.mean(features_scaled[:, start:end], axis=1)
    groups = list(FEATURE_GROUPS.keys())
    r2_matrix = np.zeros((len(groups), len(groups)))
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i != j:
                r_val = np.corrcoef(group_data[group1], group_data[group2])[0,1]
                r2_matrix[i,j] = r_val**2
    plt.figure(figsize=(8, 6))
    sns.heatmap(r2_matrix, annot=True, fmt=".3f", cmap="coolwarm", xticklabels=groups, yticklabels=groups)
    plt.title("Feature Group Correlations (R²)")
    plt.tight_layout()
    plt.savefig('figures/feature_correlation_heatmap.png')
    plt.close()

    print("Analysis figures saved in 'figures/' directory.")

def reduce_noise(audio, sr):
    """Apply noise reduction to audio signal"""
    # Apply spectral gating
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return reduced_noise

def apply_bandpass_filter(audio, sr, lowcut=80, highcut=3000):
    """Apply bandpass filter to focus on speech frequencies"""
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, audio)

def extract_enhanced_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)
        
        # Apply noise reduction and filtering
        y = reduce_noise(y, sr)
        y = apply_bandpass_filter(y, sr)
        
        target_samples_per_chunk = SAMPLE_RATE * MAX_AUDIO_LENGTH
        all_feature_chunks = []
        
        if len(y) == 0:
            print(f"Warning: Empty audio data for {file_path}, skipping feature extraction.")
            return all_feature_chunks

        num_total_samples = len(y)
        num_chunks = int(np.ceil(num_total_samples / target_samples_per_chunk))
        
        if num_chunks == 0 and num_total_samples > 0:
            num_chunks = 1
        elif num_chunks == 0 and num_total_samples == 0:
            return all_feature_chunks

        for i in range(num_chunks):
            start_sample = i * target_samples_per_chunk
            end_sample = start_sample + target_samples_per_chunk
            segment = y[start_sample:end_sample]
            
            if len(segment) < target_samples_per_chunk:
                padding_needed = target_samples_per_chunk - len(segment)
                segment = np.pad(segment, (0, padding_needed), mode='constant')
            elif len(segment) > target_samples_per_chunk:
                segment = segment[:target_samples_per_chunk]

            # Basic features
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel)
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
            
            # Ensure all features are 1D arrays before concatenation
            feature_arrays = [
                np.mean(mfcc, axis=1),                    # 13 features
                np.mean(mel_db, axis=1),                 # 128 features
                np.mean(chroma, axis=1),                 # 12 features
                np.mean(contrast, axis=1),               # 7 features
                np.mean(tonnetz, axis=1),                # 6 features
            ]
            
            # Ensure all arrays are 1D and have the correct shape
            feature_arrays = [np.atleast_1d(arr) for arr in feature_arrays]
            
            # Concatenate all features
            current_chunk_features = np.concatenate(feature_arrays)
            current_chunk_features = np.nan_to_num(current_chunk_features, nan=0.0, posinf=0.0, neginf=0.0)
            all_feature_chunks.append(current_chunk_features)
            
        return all_feature_chunks
        
    except Exception as e:
        print(f"Error processing {file_path} during feature extraction: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Gender Recognition: Train or Predict")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        choices=["train", "predict"],
        help="Operation mode: 'train' to train models, 'predict' to load and predict."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="audio",
        help="Directory containing audio files for prediction (used in predict mode, or for post-training evaluation in train mode)."
    )
    args = parser.parse_args()

    # Ensure MODEL_SAVE_DIR exists, critical for both modes if functions are called directly
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) 
    
    if args.mode == "train":
        perform_training_and_evaluation(args.audio_dir)
    elif args.mode == "predict":
        perform_prediction_with_saved_models(args.audio_dir)