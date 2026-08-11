import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    Reshape, Multiply, Lambda, Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate
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

class CBAMSENetFusionLayer(Layer):
    def __init__(self, reduction_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.reduced_dim = self.channels // self.reduction_ratio

        self.cbam_avg_pool = GlobalAveragePooling2D()
        self.cbam_max_pool = GlobalMaxPooling2D()
        self.cbam_dense1 = Dense(self.reduced_dim, activation='relu', kernel_initializer='he_normal')
        self.cbam_dense2 = Dense(self.channels, activation='sigmoid', kernel_initializer='he_normal')

        self.senet_avg_pool = GlobalAveragePooling2D()
        self.senet_dense1 = Dense(self.reduced_dim, activation='relu', kernel_initializer='he_normal')
        self.senet_dense2 = Dense(self.channels, activation='sigmoid', kernel_initializer='he_normal')
        super().build(input_shape)

    def call(self, inputs, training=None):
        cbam_avg = self.cbam_avg_pool(inputs)
        cbam_max = self.cbam_max_pool(inputs)
        cbam_concat = Concatenate(axis=-1)([cbam_avg, cbam_max])
        cbam_fc1 = self.cbam_dense1(cbam_concat)
        cbam_fc2 = self.cbam_dense2(cbam_fc1)
        cbam_weight = Reshape((1, 1, self.channels))(cbam_fc2)
        cbam_output = Multiply()([inputs, cbam_weight])

        senet_avg = self.senet_avg_pool(cbam_output)
        senet_fc1 = self.senet_dense1(senet_avg)
        senet_fc2 = self.senet_dense2(senet_fc1)
        senet_weight = Reshape((1, 1, self.channels))(senet_fc2)
        fusion_output = Multiply()([cbam_output, senet_weight])
        return fusion_output

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
    y = np.where(y == unique_labels[1], 1, 0)
    return X_68d, y, df[id_col].values

def fuse_features():
    X_6d_train, y_6d_train = load_6d_data("train_6.csv")
    X_68d_train, y_68d_train, _ = load_68d_data("train_phys_68_standardized.csv")
    X_6d_val, y_6d_val = load_6d_data("val_6.csv")
    X_68d_val, y_68d_val, val_ids = load_68d_data("val_phys_68_standardized.csv")

    X_train_fuse = np.concatenate([X_6d_train, X_68d_train], axis=1)
    X_val_fuse = np.concatenate([X_6d_val, X_68d_val], axis=1)

    X_train = X_train_fuse.reshape(-1, 74, 1, 1)
    X_val = X_val_fuse.reshape(-1, 74, 1, 1)

    y_train = to_categorical(y_68d_train, 2)
    y_val = y_68d_val
    y_val_onehot = to_categorical(y_val, 2)

    return X_train, y_train, X_val, y_val, y_val_onehot, val_ids

X_train, y_train, X_val, y_val, y_val_onehot, val_ids = fuse_features()

def build_fusion_attention_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(64, (5, 1), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)

    x = Conv2D(128, (5, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)

    x = CBAMSENetFusionLayer(reduction_ratio=2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_fusion_attention_model((74, 1, 1))
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val_onehot),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

def calculate_metrics(y_true, y_pred, y_pred_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'ACC': round(accuracy_score(y_true, y_pred),4),
        'Sn': round(tp/(tp+fn) if (tp+fn)!=0 else 0,4),
        'Sp': round(tn/(tn+fp) if (tn+fp)!=0 else 0,4),
        'MCC': round(matthews_corrcoef(y_true, y_pred),4),
        'F1': round(f1_score(y_true, y_pred),4),
        'AUC': round(roc_auc_score(y_true, y_pred_prob[:,1]),4),
        'AUPR': round(auc(*precision_recall_curve(y_true, y_pred_prob[:,1])[1::-1]),4)
    }

y_val_pred_prob = model.predict(X_val, verbose=0)
y_val_pred = np.argmax(y_val_pred_prob, axis=1)
val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_prob)

with open(os.path.join(output_dir, "CNN+SENet+CBAM_val_metrics.json"), 'w', encoding='utf-8') as f:
    json.dump({"val_metrics": val_metrics}, f, ensure_ascii=False, indent=4)

pd.DataFrame({
    "protein_id": val_ids,
    "true_label": y_val,
    "pred_label": y_val_pred,
    "pred_prob_class1": y_val_pred_prob[:,1]
}).to_csv(os.path.join(output_dir, "SENet_CBAM_val_predictions.csv"), index=False, encoding='gbk')