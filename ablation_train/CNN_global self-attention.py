import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    Reshape, Multiply, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, \
    precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

DATA_DIR_6D = "./data/raw"
DATA_DIR_68D = "./data/standardized"
output_dir = "./output/zhuyili"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class GlobalSelfAttentionLayer(Layer):
    def __init__(self, d_k=64, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.seq_len = input_shape[1]
        self.q_dense = Dense(self.d_k, activation='relu', kernel_initializer='he_normal')
        self.k_dense = Dense(self.d_k, activation='relu', kernel_initializer='he_normal')
        self.v_dense = Dense(self.channels, activation='relu', kernel_initializer='he_normal')
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = tf.squeeze(inputs, axis=2)
        Q = self.q_dense(x)
        K = self.k_dense(x)
        V = self.v_dense(x)
        K_T = tf.transpose(K, perm=[0, 2, 1])
        attention_score = tf.matmul(Q, K_T)
        scale = 1.0 / tf.sqrt(tf.cast(self.d_k, tf.float32))
        attention_score = attention_score * scale
        attention_weight = tf.nn.softmax(attention_score, axis=-1)
        attention_output = tf.matmul(attention_weight, V)
        attention_output = tf.expand_dims(attention_output, axis=2)
        output = inputs + attention_output
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

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
        raise ValueError(f"Label count is not 2: {unique_labels}")
    if not np.all(np.isin(y, [0, 1])):
        y = np.where(y == unique_labels[1], 1, 0)
    return X_68d, y, df[id_col].values

def fuse_features():
    X_6d_train, y_6d_train = load_6d_data("train_6.csv")
    X_6d_val, y_6d_val = load_6d_data("val_6.csv")
    X_68d_train, y_68d_train, train_ids = load_68d_data("train_phys_68_standardized.csv")
    X_68d_val, y_68d_val, val_ids = load_68d_data("val_phys_68_standardized.csv")
    assert X_6d_train.shape[0] == X_68d_train.shape[0]
    assert X_6d_val.shape[0] == X_68d_val.shape[0]
    assert np.all(y_6d_train == y_68d_train)
    assert np.all(y_6d_val == y_68d_val)
    X_train_fuse = np.concatenate([X_6d_train, X_68d_train], axis=1)
    X_val_fuse = np.concatenate([X_6d_val, X_68d_val], axis=1)
    X_train = X_train_fuse.reshape(-1, 74, 1, 1)
    X_val = X_val_fuse.reshape(-1, 74, 1, 1)
    y_train = y_68d_train
    y_val = y_68d_val
    y_train_onehot = to_categorical(y_train, num_classes=2)
    y_val_onehot = to_categorical(y_val, num_classes=2)
    return (X_train, y_train, y_train_onehot,
            X_val, y_val, y_val_onehot,
            train_ids, val_ids)

(X_train, y_train, y_train_onehot,
 X_val, y_val, y_val_onehot,
 train_ids, val_ids) = fuse_features()

def build_cnn_self_attention(params, input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(
        filters=params['filters'],
        kernel_size=(params['kernel_size'], 1),
        activation='relu',
        padding='same'
    )(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(params['pool_size'], 1))(x)
    x = Conv2D(
        filters=params['filters'] * 2,
        kernel_size=(params['kernel_size'], 1),
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(params['pool_size'], 1))(x)
    x = GlobalSelfAttentionLayer(d_k=64)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs, name="CNN_74D_SelfAttention")
    optimizer = Adam(learning_rate=params['lr'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_best_params():
    original_params_path = "./output/fusion_6_68/best_hyperparameters.json"
    if os.path.exists(original_params_path):
        with open(original_params_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        return best_params
    else:
        raise FileNotFoundError(f"Best params file not found: {original_params_path}")

best_params = load_best_params()

def train_cnn_self_attention():
    input_shape = (74, 1, 1)
    model = build_cnn_self_attention(best_params, input_shape)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    return model, history

self_attn_model, self_attn_history = train_cnn_self_attention()

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
    auc_pr = auc(recall, precision)
    return {
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'ACC': round(acc, 4), 'Sn': round(sn, 4), 'Sp': round(sp, 4),
        'MCC': round(mcc, 4), 'F1': round(f1, 4),
        'AUC': round(auc_roc, 4), 'AUPR': round(auc_pr, 4)
    }

y_val_pred_prob = self_attn_model.predict(X_val, verbose=0)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)
val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_prob)

model_info = {
    "model_type": "CNN+SelfAttention",
    "base_params_source": "original_74d_fusion_best_params",
    "attention_params": {"d_k": 64}
}
with open(os.path.join(output_dir, "CNN+SelfAttention_model_info.json"), 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=4)

with open(os.path.join(output_dir, "CNN+SelfAttention_val_metrics.json"), 'w', encoding='utf-8') as f:
    json.dump({"val_metrics": val_metrics}, f, ensure_ascii=False, indent=4)

def save_pred_prob(ids, y_true, y_pred, y_pred_prob, save_name):
    df = pd.DataFrame({
        "protein_id": ids,
        "true_label": y_true,
        "pred_label": y_pred,
        "adhesion_prob": y_pred_prob[:, 1],
        "non_adhesion_prob": y_pred_prob[:, 0]
    })
    df.to_csv(os.path.join(output_dir, save_name), index=False, encoding="gbk")

save_pred_prob(val_ids, y_val, y_val_pred, y_val_pred_prob, "CNN+SelfAttention_val_predictions.csv")

self_attn_model.save(os.path.join(output_dir, "CNN+SelfAttention_best_model.h5"))