import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling2D, Reshape, Multiply, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, \
    precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

DATA_DIR_6D = "./data/raw"
DATA_DIR_68D = "./data/standardized"
output_dir = "./output/zhuyili"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def se_block(input_feature, ratio=2):
    channel = input_feature.shape[-1]
    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    se_feature = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se_feature)
    se_feature = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se_feature)
    return Multiply()([input_feature, se_feature])

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

def fuse_features():
    X6d_tr, y6d_tr = load_6d_data("train_6.csv")
    X68d_tr, y68d_tr, tr_ids = load_68d_data("train_phys_68_standardized.csv")
    X6d_val, y6d_val = load_6d_data("val_6.csv")
    X68d_val, y68d_val, val_ids = load_68d_data("val_phys_68_standardized.csv")
    X6d_te, y6d_te = load_6d_data("test_6.csv")
    X68d_te, y68d_te, te_ids = load_68d_data("test_phys_68_standardized.csv")

    X_tr = np.concatenate([X6d_tr, X68d_tr], axis=1).reshape(-1,74,1,1)
    X_val = np.concatenate([X6d_val, X68d_val], axis=1).reshape(-1,74,1,1)
    X_te = np.concatenate([X6d_te, X68d_te], axis=1).reshape(-1,74,1,1)

    y_tr = to_categorical(y68d_tr, 2)
    y_val = to_categorical(y68d_val, 2)
    y_te = to_categorical(y68d_te, 2)
    return X_tr, y_tr, X_val, y_val, X_te, y_te, y68d_val, val_ids

X_train, y_train, X_val, y_val_onehot, X_test, y_test_onehot, y_val, val_ids = fuse_features()

def build_cnn_se_global(params, input_shape=(74,1,1)):
    inp = tf.keras.Input(shape=input_shape)

    x = Conv2D(params['filters'], (params['kernel_size'],1), activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((params['pool_size'],1))(x)

    x = Conv2D(params['filters']*2, (params['kernel_size'],1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((params['pool_size'],1))(x)

    x = se_block(x)
    x = GlobalSelfAttentionLayer(d_k=64)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(params['dropout_rate'])(x)
    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(Adam(learning_rate=params['lr']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_best_params():
    path = "./output/fusion_6_68/best_hyperparameters.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

best_params = load_best_params()

model = build_cnn_se_global(best_params)
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val_onehot),
          epochs=50, batch_size=32, callbacks=[es], verbose=0)

def calculate_metrics(y_true, y_pred, y_pred_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sn = tp/(tp+fn) if (tp+fn)!=0 else 0
    sp = tn/(tn+fp) if (tn+fp)!=0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    y_prob = y_pred_prob[:,1]
    auc_roc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    sorted_idx = np.argsort(recall)
    auc_pr = auc(recall[sorted_idx], precision[sorted_idx])
    return {
        'ACC': round(acc,4),
        'Sn': round(sn,4),
        'Sp': round(sp,4),
        'MCC': round(mcc,4),
        'F1': round(f1,4),
        'AUC': round(auc_roc,4),
        'AUPR': round(auc_pr,4)
    }

y_val_pred_prob = model.predict(X_val, verbose=0)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)
val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_prob)

with open(os.path.join(output_dir, "CNN+SENet+GlobalSelfAttention_val_metrics.json"), 'w', encoding='utf-8') as f:
    json.dump({"val_metrics": val_metrics}, f, ensure_ascii=False, indent=4)