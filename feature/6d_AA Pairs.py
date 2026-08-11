import os
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, \
    Reshape, Multiply, Lambda, Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, \
    precision_recall_curve, auc, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

DATA_FASTA_DIR = "./data/fasta"
DATA_CSV_DIR = "./data/csv"
OUTPUT_DIR = "./output"
os.makedirs(DATA_FASTA_DIR, exist_ok=True)
os.makedirs(DATA_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def merge_fasta(file1, file2, output_merged):
    seen_ids = set()
    with open(output_merged, 'w', encoding='utf-8') as out_f:
        with open(file1, 'r', encoding='utf-8') as f1:
            for line in f1:
                if line.startswith(">"):
                    seq_id = line.strip()
                    if seq_id not in seen_ids:
                        seen_ids.add(seq_id)
                        out_f.write(line)
                    else:
                        continue
                else:
                    out_f.write(line)
        with open(file2, 'r', encoding='utf-8') as f2:
            for line in f2:
                if line.startswith(">"):
                    seq_id = line.strip()
                    if seq_id not in seen_ids:
                        seen_ids.add(seq_id)
                        out_f.write(line)
                    else:
                        continue
                else:
                    out_f.write(line)

def encode_amino_acids(input_fasta, output_encoded):
    def encode_seq(seq):
        seq = seq.upper()
        seq = seq.replace("G", "").replace("A", "").replace("P", "")
        seq = seq.translate(str.maketrans("VLIM", "VVVV"))
        seq = seq.translate(str.maketrans("STC", "SSS"))
        seq = seq.translate(str.maketrans("DE", "DD"))
        seq = seq.translate(str.maketrans("HKR", "HHH"))
        seq = seq.translate(str.maketrans("NQ", "NN"))
        seq = seq.translate(str.maketrans("FYW", "FFF"))
        return seq

    with open(input_fasta, 'r', encoding='utf-8') as in_f, \
         open(output_encoded, 'w', encoding='utf-8') as out_f:
        current_id = ""
        current_seq = ""
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id and current_seq:
                    encoded_seq = encode_seq(current_seq)
                    out_f.write(f"{current_id}\n{encoded_seq}\n")
                current_id = line
                current_seq = ""
            else:
                current_seq += line
        if current_id and current_seq:
            encoded_seq = encode_seq(current_seq)
            out_f.write(f"{current_id}\n{encoded_seq}\n")

def count_adjacent_frequency(encoded_fasta, original_fasta, output_csv):
    original_seq_dict = {}
    with open(original_fasta, 'r', encoding='utf-8') as f:
        current_id = ""
        current_seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id and current_seq:
                    original_seq_dict[current_id.lstrip(">").strip()] = current_seq
                current_id = line.lstrip(">").strip()
                current_seq = ""
            else:
                current_seq += line
        if current_id and current_seq:
            original_seq_dict[current_id] = current_seq

    def count_duplets(seq, duplet):
        count = 0
        if len(seq) < 2:
            return 0.0
        for i in range(len(seq)-1):
            if seq[i] == duplet[0] and seq[i+1] == duplet[1]:
                count += 1
        frequency = round(count / (len(seq)-1), 6) if len(seq) > 1 else 0.0
        return frequency

    target_duplets = ["VV", "SS", "DD", "HH", "NN", "FF"]
    results = []
    with open(encoded_fasta, 'r', encoding='utf-8') as f:
        current_id = ""
        current_encoded_seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id and current_encoded_seq:
                    pure_id = current_id.lstrip(">").strip()
                    freq_dict = {dup: count_duplets(current_encoded_seq, dup) for dup in target_duplets}
                    original_seq = original_seq_dict.get(pure_id, "")
                    results.append({
                        "protein_id": pure_id,
                        "VV_freq": freq_dict["VV"],
                        "SS_freq": freq_dict["SS"],
                        "DD_freq": freq_dict["DD"],
                        "HH_freq": freq_dict["HH"],
                        "NN_freq": freq_dict["NN"],
                        "FF_freq": freq_dict["FF"],
                        "original_sequence": original_seq
                    })
                current_id = line
                current_encoded_seq = ""
            else:
                current_encoded_seq += line
        if current_id and current_encoded_seq:
            pure_id = current_id.lstrip(">").strip()
            freq_dict = {dup: count_duplets(current_encoded_seq, dup) for dup in target_duplets}
            original_seq = original_seq_dict.get(pure_id, "")
            results.append({
                "protein_id": pure_id,
                "VV_freq": freq_dict["VV"],
                "SS_freq": freq_dict["SS"],
                "DD_freq": freq_dict["DD"],
                "HH_freq": freq_dict["HH"],
                "NN_freq": freq_dict["NN"],
                "FF_freq": freq_dict["FF"],
                "original_sequence": original_seq
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return df

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)
    sn = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    metrics_dict = {
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'ACC': round(acc, 4),
        'MCC': round(mcc, 4),
        'F1': round(f1, 4),
        'AUC': round(auc_roc, 4),
        'AUPR': round(auc_pr, 4),
        'Sn': round(sn, 4),
        'Sp': round(sp, 4)
    }
    return metrics_dict

def load_and_preprocess_data():
    train_df = pd.read_csv(os.path.join(DATA_CSV_DIR, "train_6.csv"), encoding='utf-8')
    val_df = pd.read_csv(os.path.join(DATA_CSV_DIR, "val_6.csv"), encoding='utf-8')

    feature_cols = ['VV_freq', 'SS_freq', 'DD_freq', 'HH_freq', 'NN_freq', 'FF_freq']
    X_train = train_df[feature_cols].values
    y_train = train_df['protein_label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['protein_label'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = X_train.reshape(X_train.shape[0], 6, 1)
    X_val = X_val.reshape(X_val.shape[0], 6, 1)

    return X_train, y_train, X_val, y_val

def build_cnn(filters=32, kernel_size=2, pool_size=2, dropout_rate=0.2, lr=0.001):
    model = Sequential(name="CNN_6D_Feature")
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(6, 1), name="Conv1D_Layer"))
    model.add(MaxPooling1D(pool_size=pool_size, name="MaxPool1D_Layer"))
    model.add(Dropout(rate=dropout_rate, name="Dropout_Layer"))
    model.add(Flatten(name="Flatten_Layer"))
    model.add(Dense(units=16, activation='relu', name="Dense_Layer"))
    model.add(Dense(units=1, activation='sigmoid', name="Output_Layer"))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def hyperparameter_tuning(X_train, y_train):
    model = KerasClassifier(model=build_cnn, epochs=20, batch_size=16, verbose=0)
    param_grid = {
        'model__filters': [16, 32, 64],
        'model__kernel_size': [2, 3],
        'model__pool_size': [2],
        'model__dropout_rate': [0.2, 0.3, 0.4],
        'model__lr': [0.001, 0.0001]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    grid_results_df = pd.DataFrame(grid_search.cv_results_)
    grid_results_df.to_csv(os.path.join(OUTPUT_DIR, "hyperparameter_tuning_results.csv"), index=False, encoding='utf-8')
    return best_params

def train_best_model(best_params, X_train, y_train, X_val, y_val):
    best_model = build_cnn(
        filters=best_params['filters'],
        kernel_size=best_params['kernel_size'],
        pool_size=best_params['pool_size'],
        dropout_rate=best_params['dropout_rate'],
        lr=best_params['lr']
    )
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0
    )
    best_model.save(os.path.join(OUTPUT_DIR, "best_cnn_model.h5"))
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False, encoding='utf-8')
    return best_model

def evaluate_datasets(model, X_train, y_train, X_val, y_val):
    y_train_prob = model.predict(X_train, verbose=0).flatten()
    y_train_pred = (y_train_prob >= 0.5).astype(int)
    y_val_prob = model.predict(X_val, verbose=0).flatten()
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_prob)
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)

    pd.DataFrame([train_metrics]).to_csv(os.path.join(OUTPUT_DIR, "train_metrics.csv"), index=False, encoding='utf-8')
    pd.DataFrame([val_metrics]).to_csv(os.path.join(OUTPUT_DIR, "val_metrics.csv"), index=False, encoding='utf-8')

    compare_df = pd.DataFrame({
        'metric': list(train_metrics.keys()),
        'train': list(train_metrics.values()),
        'val': list(val_metrics.values())
    })
    compare_df.to_csv(os.path.join(OUTPUT_DIR, "train_val_comparison.csv"), index=False, encoding='utf-8')

    train_pred_df = pd.DataFrame({'true_label': y_train, 'pred_label': y_train_pred, 'probability': y_train_prob})
    val_pred_df = pd.DataFrame({'true_label': y_val, 'pred_label': y_val_pred, 'probability': y_val_prob})
    train_pred_df.to_csv(os.path.join(OUTPUT_DIR, "train_predictions.csv"), index=False, encoding='utf-8')
    val_pred_df.to_csv(os.path.join(OUTPUT_DIR, "val_predictions.csv"), index=False, encoding='utf-8')

    return compare_df

if __name__ == "__main__":
    pure1_fasta = os.path.join(DATA_FASTA_DIR, "pure1.fasta")
    nian1_fasta = os.path.join(DATA_FASTA_DIR, "nian1.fasta")
    merged_fasta = os.path.join(DATA_FASTA_DIR, "merged.fasta")
    merged_encoded_fasta = os.path.join(DATA_FASTA_DIR, "merged_encoded.fasta")
    freq_csv = os.path.join(DATA_CSV_DIR, "protein_frequencies.csv")

    try:
        merge_fasta(pure1_fasta, nian1_fasta, merged_fasta)
        encode_amino_acids(merged_fasta, merged_encoded_fasta)
        count_adjacent_frequency(merged_encoded_fasta, merged_fasta, freq_csv)
    except Exception:
        pass

    X_train, y_train, X_val, y_val = load_and_preprocess_data()
    best_params = hyperparameter_tuning(X_train, y_train)
    best_model = train_best_model(best_params, X_train, y_train, X_val, y_val)
    compare_df = evaluate_datasets(best_model, X_train, y_train, X_val, y_val)