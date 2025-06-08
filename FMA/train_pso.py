import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pyswarms as ps

FMA_METADATA_PATH = 'data/raw/fma_metadata/tracks.csv'
SPECTROGRAM_PATH = 'data/processed/spectrograms/'

N_MELS = 128
N_TIME_STEPS = 1292
INPUT_SHAPE = (N_MELS, N_TIME_STEPS, 1)

def load_data(metadata_path, spectrograms_dir):
    """
    Loads spectrograms and their corresponding genre labels.
    """
    try:
        tracks_df = pd.read_csv(metadata_path, header=[0, 1], index_col=0)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}")
        return None, None

    small_subset = tracks_df[tracks_df[('set', 'subset')] == 'small']
    small_subset = small_subset[~small_subset[('track', 'genre_top')].isna()]

    spectrograms = []
    genres = []
    track_ids_to_load = small_subset.index

    print(f"Found {len(track_ids_to_load)} tracks in the 'small' subset with genre labels.")

    for track_id in tqdm(track_ids_to_load, desc="Loading spectrograms"):
        spec_path = os.path.join(spectrograms_dir, f'{track_id:06d}.npy')
        if os.path.exists(spec_path):
            spec = np.load(spec_path)
            if spec.shape[1] == N_TIME_STEPS:
                spectrograms.append(spec)
                genres.append(small_subset.loc[track_id][('track', 'genre_top')])

    X = np.array(spectrograms).reshape(-1, N_MELS, N_TIME_STEPS, 1)
    y = np.array(genres)

    return X, y

X, y = load_data(FMA_METADATA_PATH, SPECTROGRAM_PATH)

if X is None or len(X) == 0:
    print("Data loading failed or no data was found. Exiting.")
    exit()

print(f"\nLoaded {X.shape[0]} spectrograms.")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\nTarget variable 'genre' encoded.")
print(f"Number of unique classes: {num_classes}")
print(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"X_train_full shape: {X_train_full.shape}, y_train_full shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

baseline_model = Sequential([
    Input(shape=INPUT_SHAPE),

    # Block 1
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(), 
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') 
])

baseline_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
baseline_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = baseline_model.fit(
    X_train_full, y_train_full,
    epochs=50,
    batch_size=32,
    validation_split=0.2, 
    callbacks=[early_stopping],
    verbose=1
)

loss, accuracy = baseline_model.evaluate(X_test, y_test, verbose=0)
print(f"\nBaseline Test Loss: {loss:.4f}")
print(f"Baseline Test Accuracy: {accuracy:.4f}")

y_pred_probs = baseline_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nBaseline Model Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

baseline_report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, output_dict=True)
baseline_macro_f1 = baseline_report_dict.get('macro avg', {}).get('f1-score', 0.0)
baseline_weighted_f1 = baseline_report_dict.get('weighted avg', {}).get('f1-score', 0.0)
baseline_mcc = matthews_corrcoef(y_test, y_pred)
baseline_cohen_kappa = cohen_kappa_score(y_test, y_pred)
baseline_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

X_train_pso, X_val_pso, y_train_pso, y_val_pso = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=123, stratify=y_train_full
)
print(f"PSO training set size: {X_train_pso.shape}")
print(f"PSO validation set size: {X_val_pso.shape}")

min_bounds = np.array([16, 32, 64,  0.1,  0.0001])
max_bounds = np.array([64, 128, 256, 0.5,  0.01])
bounds_pso = (min_bounds, max_bounds)
n_dimensions = len(min_bounds)

pso_iteration_count = 0
def cnn_fitness_function(params):
    """Builds, trains, and evaluates a CNN for one particle."""
    global pso_iteration_count
    pso_iteration_count += 1

    filters_conv1 = int(round(params[0]))
    filters_conv2 = int(round(params[1]))
    dense_units = int(round(params[2]))
    dropout_rate = float(params[3])
    learning_rate = float(params[4])

    print(f"\nPSO Eval {pso_iteration_count}: F1={filters_conv1}, F2={filters_conv2}, Dense={dense_units}, DR={dropout_rate:.3f}, LR={learning_rate:.5f}")

    tf.keras.backend.clear_session()

    model_pso = Sequential([
        Input(shape=INPUT_SHAPE),
        Conv2D(filters_conv1, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),

        Conv2D(filters_conv2, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),

        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model_pso.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop_pso = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_pso = model_pso.fit(
        X_train_pso, y_train_pso,
        epochs=25, 
        batch_size=32, 
        validation_data=(X_val_pso, y_val_pso),
        callbacks=[early_stop_pso],
        verbose=0 
    )

    val_accuracy = np.max(history_pso.history.get('val_accuracy', [0]))
    print(f" -> Best Val Acc: {val_accuracy:.4f}")

    return 1.0 - val_accuracy

def pso_objective_wrapper(particles_batch):
    """Wrapper to evaluate a batch of particles."""
    return np.array([cnn_fitness_function(p) for p in particles_batch])

n_particles_pso = 8
pso_iters = 10
options_pso = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

print(f"\nStarting PSO with {n_particles_pso} particles for {pso_iters} iterations.")
print("This will take a significant amount of time...")

pso_optimizer = ps.single.GlobalBestPSO(
    n_particles=n_particles_pso,
    dimensions=n_dimensions,
    options=options_pso,
    bounds=bounds_pso
)

best_cost_pso, best_params_pso = pso_optimizer.optimize(pso_objective_wrapper, iters=pso_iters, verbose=True)

print("\n--- PSO Optimization Finished ---")
print(f"Best cost (1 - val_accuracy) from PSO: {best_cost_pso:.4f}")
print(f"Best hyperparameters from PSO: {best_params_pso}")

best_filters_conv1 = int(round(best_params_pso[0]))
best_filters_conv2 = int(round(best_params_pso[1]))
best_dense_units = int(round(best_params_pso[2]))
best_dropout_rate = float(best_params_pso[3])
best_learning_rate = float(best_params_pso[4])

print("\nFormatted Best Hyperparameters from PSO:")
print(f"  Filters Conv1: {best_filters_conv1}, Filters Conv2: {best_filters_conv2}")
print(f"  Dense Units: {best_dense_units}, Dropout: {best_dropout_rate:.4f}, LR: {best_learning_rate:.6f}")

print("\n--- Training Final Model with PSO-Optimized Hyperparameters ---")
tf.keras.backend.clear_session()
final_model_pso = Sequential([
    Input(shape=INPUT_SHAPE),
    Conv2D(best_filters_conv1, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(best_dropout_rate),
    Conv2D(best_filters_conv2, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(best_dropout_rate),
    Flatten(),
    Dense(best_dense_units, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
final_optimizer = Adam(learning_rate=best_learning_rate)
final_model_pso.compile(optimizer=final_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model_pso.summary()

early_stopping_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_final_pso = final_model_pso.fit(
    X_train_full, y_train_full,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping_final],
    verbose=1
)

print("\n--- Evaluating PSO Optimized Model on Test Set ---")
loss_final_pso, accuracy_final_pso = final_model_pso.evaluate(X_test, y_test, verbose=0)
print(f"\nPSO Optimized Test Loss: {loss_final_pso:.4f}")
print(f"PSO Optimized Test Accuracy: {accuracy_final_pso:.4f}")

y_pred_probs_final_pso = final_model_pso.predict(X_test)
y_pred_final_pso = np.argmax(y_pred_probs_final_pso, axis=1)

print("\nPSO Optimized Model Classification Report:")
print(classification_report(y_test, y_pred_final_pso, target_names=label_encoder.classes_, zero_division=0))

pso_report_dict = classification_report(y_test, y_pred_final_pso, target_names=label_encoder.classes_, zero_division=0, output_dict=True)
pso_macro_f1 = pso_report_dict.get('macro avg', {}).get('f1-score', 0.0)
pso_weighted_f1 = pso_report_dict.get('weighted avg', {}).get('f1-score', 0.0)
pso_mcc = matthews_corrcoef(y_test, y_pred_final_pso)
pso_cohen_kappa = cohen_kappa_score(y_test, y_pred_final_pso)
pso_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_final_pso)

print("\n--- Comparison of Model Performance ---")
comparison_data = {
    "Metric": ["Accuracy", "Macro F1-score", "Weighted F1-score", "Balanced Accuracy", "MCC", "Cohen's Kappa"],
    "Baseline Model": [accuracy, baseline_macro_f1, baseline_weighted_f1, baseline_balanced_accuracy, baseline_mcc, baseline_cohen_kappa],
    "PSO Optimized Model": [accuracy_final_pso, pso_macro_f1, pso_weighted_f1, pso_balanced_accuracy, pso_mcc, pso_cohen_kappa]
}
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False, float_format="%.4f"))

plt.figure(figsize=(18, 7))

plt.subplot(1, 2, 1)
cm_baseline = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Baseline Model Confusion Matrix')
plt.xlabel('Predicted Label'); plt.ylabel('True Label')

plt.subplot(1, 2, 2)
cm_pso = confusion_matrix(y_test, y_pred_final_pso)
sns.heatmap(cm_pso, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('PSO Optimized Model Confusion Matrix')
plt.xlabel('Predicted Label'); plt.ylabel('True Label')

plt.tight_layout()
plt.show()