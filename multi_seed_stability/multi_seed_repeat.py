import pandas as pd
import numpy as np
import os
import json
import random
import warnings
warnings.filterwarnings('ignore')

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

DATA_DIR_6D = r"data/raw"
DATA_DIR_68D = r"data/raw/standardized"
output_dir = r"outputs/multi_seed_output"
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

class SelfAttentionLayer(Layer):
    def __init__(self, dim=64, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
    def build(self, input_shape):
        channels = input_shape[-1]
        self.q = Dense(self.dim)
        self.k = Dense(self.dim)
        self.v = Dense(channels)
    def call(self, x):
        t = tf.squeeze(x, axis=2)
        q = self.q(t)
        k = self.k(t)
        v = self.v(t)
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.dim)))
        out = tf.matmul(attn, v)
        out = tf.expand_dims(out, axis=2)
        return x + out

def load_6d_data(file_name):
    file_path = os.path.join(DATA_DIR_6D, file_name)
    df = pd.read_csv(file_path, encoding='utf-8')
    feature_cols_6d = ['VV_frequency', 'SS_frequency', 'DD_frequency', 'HH_frequency', 'NN_frequency', 'FF_frequency']
    X_6d = df[feature_cols_6d].values
    y = df['protein_type_label'].values
    scaler = StandardScaler()
    X_6d = scaler.fit_transform(X_6d)
    return X_6d, y

def load_68d_data(file_name):
    file_path = os.path.join(DATA_DIR_68D, file_name)
    df = pd.read_csv(file_path, encoding="gbk")
    id_col = "protein_id"
    label_col = "protein_type_label"
    feature_cols_68d = [col for col in df.columns if col not in [id_col, label_col]]
    X_68d = df[feature_cols_68d].values
    y = df[label_col].values
    unique_labels = np.unique(y)
    y = np.where(y == unique_labels[1], 1, 0)
    return X_68d, y, df[id_col].values

def fuse_features():
    X_6d_train, y_6d_train = load_6d_data("train_6.csv")
    X_68d_train, y_68d_train, train_ids = load_68d_data("train_phys_68_standardized.csv")
    X_6d_val, y_6d_val = load_6d_data("val_6.csv")
    X_68d_val, y_68d_val, val_ids = load_68d_data("val_phys_68_standardized.csv")
    X_6d_test, y_6d_test = load_6d_data("test_6.csv")
    X_68d_test, y_68d_test, test_ids = load_68d_data("test_phys_68_standardized.csv")

    X_train_fuse = np.concatenate([X_6d_train, X_68d_train], axis=1)
    X_val_fuse = np.concatenate([X_6d_val, X_68d_val], axis=1)
    X_test_fuse = np.concatenate([X_6d_test, X_68d_test], axis=1)

    X_train = X_train_fuse.reshape(-1, 74, 1, 1)
    X_val = X_val_fuse.reshape(-1, 74, 1, 1)
    X_test = X_test_fuse.reshape(-1, 74, 1, 1)

    y_train = y_68d_train
    y_val = y_68d_val
    y_test = y_68d_test

    y_train_onehot = to_categorical(y_train, num_classes=2)
    y_val_onehot = to_categorical(y_val, num_classes=2)
    y_test_onehot = to_categorical(y_test, num_classes=2)

    return (X_train, y_train, y_train_onehot,
            X_val, y_val, y_val_onehot,
            X_test, y_test, y_test_onehot)

def build_fusion_attention_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(64, (5,1), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,1))(x)
    x = Conv2D(128, (5,1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,1))(x)

    x = SelfAttentionLayer(64)(x)
    x = CBAMSENetFusionLayer(reduction_ratio=2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0007), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
        'ACC': round(acc, 4), 'Sn': round(sn, 4), 'Sp': round(sp, 4),
        'MCC': round(mcc, 4), 'F1': round(f1, 4),
        'AUC': round(auc_roc, 4), 'AUPR': round(auc_pr, 4)
    }

def main_multi_seed_run():
    seed_list = [12, 34, 56, 78, 90]
    repeat_result_list = []
    input_shape = (74,1,1)

    for idx, current_seed in enumerate(seed_list):
        print(f"\n>>>>>>>>>>>>>> Starting run {idx+1}/5, current seed = {current_seed}")
        random.seed(current_seed)
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)

        (X_train, y_train, y_train_onehot,
         X_val, y_val, y_val_onehot,
         X_test, y_test, y_test_onehot) = fuse_features()

        model = build_fusion_attention_model(input_shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        _ = model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            epochs=50, batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        y_test_pred_prob = model.predict(X_test, verbose=0)
        y_test_pred = np.argmax(y_test_pred_prob, axis=1)
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_prob)

        run_item = {
            "seed": current_seed,
            "test_metrics": test_metrics
        }
        repeat_result_list.append(run_item)
        print(f"<<<<<<<<<<<<<< seed {current_seed} finished, test metrics: {test_metrics}")

    out_json_path = os.path.join(output_dir, "multi_seed_test_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(repeat_result_list, f, ensure_ascii=False, indent=4)
    print(f"\nAll repeated runs finished. Raw results saved to: {out_json_path}")

    metric_names = ["ACC","Sn","Sp","MCC","F1","AUC","AUPR"]
    all_values = {m:[] for m in metric_names}
    for entry in repeat_result_list:
        tm = entry["test_metrics"]
        for m in metric_names:
            all_values[m].append(tm[m])

    stat_result = {}
    for m in metric_names:
        arr = np.array(all_values[m])
        stat_result[m] = {
            "mean": round(float(np.mean(arr)),4),
            "std": round(float(np.std(arr)),4)
        }

    stat_out_path = os.path.join(output_dir, "multi_seed_stat_mean_std.json")
    with open(stat_out_path,"w",encoding="utf-8") as f:
        json.dump(stat_result,f,ensure_ascii=False,indent=4)
    print(f"\nMean and std saved to: {stat_out_path}")

    print("\n" + "="*65)
    print("==== 5-seed repeated runs, independent test set: mean ± std ====")
    for k,v in stat_result.items():
        print(f"{k:6s}: {v['mean']:.4f} ± {v['std']:.4f}")
    print("="*65)

if __name__ == "__main__":
    main_multi_seed_run()