import pandas as pd
import numpy as np
import os
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Reshape, Multiply

from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, \
    precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

DATA_DIR_6D = "./data/raw"
DATA_DIR_68D = "./data/standardized"
output_dir = "./output/zhuyili"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = "./output/zhuyili/CNN_triple_attention_model.h5"

class GlobalSelfAttentionLayer(Layer):
    def __init__(self, d_k=64, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.q_dense = Dense(self.d_k, activation='relu', kernel_initializer='he_normal')
        self.k_dense = Dense(self.d_k, activation='relu', kernel_initializer='he_normal')
        self.v_dense = Dense(self.channels, activation='relu', kernel_initializer='he_normal')
        super().build(input_shape)

    def call(self, inputs):
        x = tf.squeeze(inputs, axis=2)
        Q = self.q_dense(x)
        K = self.k_dense(x)
        V = self.v_dense(x)
        K_T = tf.transpose(K, [0, 2, 1])
        attn_score = tf.matmul(Q, K_T) / tf.sqrt(tf.cast(self.d_k, tf.float32))
        attn_weight = tf.nn.softmax(attn_score, axis=-1)
        out = tf.matmul(attn_weight, V)
        out = tf.expand_dims(out, axis=2)
        return inputs + out

def load_6d_data(file_name):
    df = pd.read_csv(os.path.join(DATA_DIR_6D, file_name), encoding='utf-8')
    cols = ['VV_freq', 'SS_freq', 'DD_freq', 'HH_freq', 'NN_freq', 'FF_freq']
    X = df[cols].values
    y = df['protein_label'].values
    X = StandardScaler().fit_transform(X)
    return X, y

def load_68d_data(file_name):
    df = pd.read_csv(os.path.join(DATA_DIR_68D, file_name), encoding="gbk")
    id_col = "protein_id"
    label_col = "protein_label"
    feat_cols = [c for c in df.columns if c not in [id_col, label_col]]
    X = df[feat_cols].values
    y = df[label_col].values
    y = np.where(y == np.unique(y)[1], 1, 0)
    return X, y, df[id_col].values

def fuse_test_features():
    X6d_te, _ = load_6d_data("test_6.csv")
    X68d_te, y68d_te, te_ids = load_68d_data("test_phys_68_standardized.csv")

    X_te = np.concatenate([X6d_te, X68d_te], axis=1).reshape(-1, 74, 1, 1)

    return X_te, y68d_te, te_ids

X_test, y_test, test_ids = fuse_test_features()

with tf.keras.utils.custom_object_scope({
    "GlobalSelfAttentionLayer": GlobalSelfAttentionLayer
}):
    model = load_model(model_path, compile=False)

y_test_pred_prob = model.predict(X_test, verbose=0)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

def calculate_metrics(y_true, y_pred, y_pred_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    sn = tp / (tp + fn) if (tp + fn) != 0 else 0
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    y_pred_prob_1 = y_pred_prob[:, 1]
    auc_roc = roc_auc_score(y_true, y_pred_prob_1)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob_1)
    sorted_idx = np.argsort(recall)
    recall_sorted = recall[sorted_idx]
    precision_sorted = precision[sorted_idx]
    auc_pr = auc(recall_sorted, precision_sorted)

    return {
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'ACC': round(acc, 4),
        'Sn': round(sn, 4),
        'Sp': round(sp, 4),
        'MCC': round(mcc, 4),
        'F1': round(f1, 4),
        'AUC': round(auc_roc, 4),
        'AUPR': round(auc_pr, 4)
    }

test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_prob)

with open(os.path.join(output_dir, "CNN_triple_attention_test_metrics.json"), 'w', encoding='utf-8') as f:
    json.dump({"test_metrics": test_metrics}, f, ensure_ascii=False, indent=4)

df_results = pd.DataFrame({
    "protein_id": test_ids,
    "true_label": y_test,
    "pred_label": y_test_pred,
    "pred_prob_class0": np.round(y_test_pred_prob[:, 0], 4),
    "pred_prob_class1": np.round(y_test_pred_prob[:, 1], 4)
})
df_results.to_csv(os.path.join(output_dir, "CNN_triple_attention_test_predictions.csv"), index=False, encoding='gbk')