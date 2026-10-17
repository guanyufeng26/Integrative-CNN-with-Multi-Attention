import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR_6D = "./data/raw"
DATA_DIR_68D = "./data/standardized"
DATA_DIR_PCA = "./data/pca"
output_dir = "./output/comparison"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_6d_data(file_name):
    file_path = os.path.join(DATA_DIR_6D, file_name)
    df = pd.read_csv(file_path, encoding='utf-8')
    feature_cols_6d = ['VV_freq', 'SS_freq', 'DD_freq', 'HH_freq', 'NN_freq', 'FF_freq']
    X_6d = df[feature_cols_6d].values
    y = df['protein_label'].values
    scaler = StandardScaler()
    X_6d = scaler.fit_transform(X_6d)
    return X_6d, y

def load_68d_data(file_name):
    file_path = os.path.join(DATA_DIR_68D, file_name)
    df = pd.read_csv(file_path, encoding="gbk")
    id_col = "protein_id"
    label_col = "protein_label"
    feature_cols_68d = [col for col in df.columns if col not in [id_col, label_col]]
    X_68d = df[feature_cols_68d].values
    y = df[label_col].values
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("Label count is not 2")
    if not np.all(np.isin(y, [0, 1])):
        y = np.where(y == unique_labels[1], 1, 0)
    return X_68d, y, df[id_col].values

def load_pca256_data(feat_path, scaler=None, pca_model=None, is_train=False):
    df = pd.read_csv(feat_path, encoding="utf-8")
    feat_cols = [col for col in df.columns if col.startswith("feat_") and "freq" not in col]
    X_pca = df[feat_cols].values.astype(np.float32)
    if is_train:
        scaler = StandardScaler()
        X_pca = scaler.fit_transform(X_pca)
        pca_model = PCA(n_components=256, random_state=42)
        X_pca = pca_model.fit_transform(X_pca)
    else:
        X_pca = scaler.transform(X_pca)
        X_pca = pca_model.transform(X_pca)
    return X_pca, scaler, pca_model

def fuse_3_features():
    X_6d_train, y_train = load_6d_data("train_6.csv")
    X_6d_val, y_val = load_6d_data("val_6.csv")
    X_68d_train, y_train_68, train_ids = load_68d_data("train_phys_68_standardized.csv")
    X_68d_val, y_val_68, val_ids = load_68d_data("val_phys_68_standardized.csv")
    X_pca_train, scaler_pca, pca_model = load_pca256_data(os.path.join(DATA_DIR_PCA, "train_features.csv"), is_train=True)
    X_pca_val, _, _ = load_pca256_data(os.path.join(DATA_DIR_PCA, "val_features.csv"), scaler=scaler_pca, pca_model=pca_model)
    assert np.all(y_train == y_train_68)
    assert np.all(y_val == y_val_68)
    X_train_fuse = np.concatenate([X_6d_train, X_68d_train, X_pca_train], axis=1)
    X_val_fuse = np.concatenate([X_6d_val, X_68d_val, X_pca_val], axis=1)
    X_train = X_train_fuse.reshape(-1, 330, 1, 1)
    X_val = X_val_fuse.reshape(-1, 330, 1, 1)
    y_train_onehot = to_categorical(y_train, 2)
    y_val_onehot = to_categorical(y_val, 2)
    return X_train, y_train, y_train_onehot, X_val, y_val, y_val_onehot, val_ids

def build_model(params, input_shape):
    model = Sequential(name="CNN_330D_Fusion")
    model.add(Conv2D(params['filters'], (params['kernel_size'], 1), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((params['pool_size'], 1)))
    model.add(Conv2D(params['filters']*2, (params['kernel_size'], 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((params['pool_size'], 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=params['lr']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def hyper_search(X_train, y_train_onehot, X_val, y_val_onehot):
    input_shape = (330, 1, 1)
    param_grid = {'filters': [32], 'kernel_size': [3], 'pool_size': [2], 'dropout_rate': [0.3], 'lr': [0.001]}
    best_f1 = 0
    best_model = None
    for params in ParameterGrid(param_grid):
        model = build_model(params, input_shape)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot),
                  epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
        y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    return best_model

def calc_metrics(y_true, y_pred, y_pred_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sn = tp/(tp+fn) if (tp+fn)!=0 else 0
    sp = tn/(tn+fp) if (tn+fp)!=0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_prob[:,1])
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:,1])
    auc_pr = auc(recall, precision)
    return {
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
        'ACC':round(acc,4),'Sn':round(sn,4),'Sp':round(sp,4),
        'MCC':round(mcc,4),'F1':round(f1,4),'AUC':round(auc_roc,4),'AUPR':round(auc_pr,4)
    }

if __name__ == "__main__":
    X_train, y_train, y_train_onehot, X_val, y_val, y_val_onehot, val_ids = fuse_3_features()
    best_model = hyper_search(X_train, y_train_onehot, X_val, y_val_onehot)
    y_val_prob = best_model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_prob, axis=1)
    val_metrics = calc_metrics(y_val, y_val_pred, y_val_prob)
    with open(os.path.join(output_dir, "6+68+256_val_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(val_metrics, f, ensure_ascii=False, indent=4)