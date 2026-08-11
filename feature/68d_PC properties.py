import os
import numpy as np
import pandas as pd
import json
import re
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import label_binarize, StandardScaler

DATA_DIR = "./data"
FASTA_DIR = os.path.join(DATA_DIR, "fasta")
CSV_DIR = os.path.join(DATA_DIR, "csv")
OUTPUT_DIR = "./output"
os.makedirs(FASTA_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_aac(seq):
    aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    total = len(seq)
    aac = [seq.count(aa)/total if total>0 else 0 for aa in aa_list]
    return aac, [f"AAC_{aa}" for aa in aa_list]

def extract_dpc(seq):
    dpc_list = ['AL','AR','KR','LL','RR','AG','GD','VE','SN','TD','HI','YK','FM','PW','TS','EV','QA','NH']
    total = len(seq)
    dpc = []
    for dp in dpc_list:
        count = 0
        for i in range(total-1):
            if seq[i]+seq[i+1] == dp:
                count +=1
        dpc.append(count/(total-1) if total>1 else 0)
    return dpc, [f"DPC_{dp}" for dp in dpc_list]

def extract_hydrophobicity(seq):
    kyte_doolittle = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    hyd_values = [kyte_doolittle[aa] for aa in seq if aa in kyte_doolittle]
    if not hyd_values:
        return [0,0,0,0], ['Hyd_mean','Hyd_std','Hyd_max','Hyd_median']
    return [np.mean(hyd_values), np.std(hyd_values), np.max(hyd_values), np.median(hyd_values)], ['Hyd_mean','Hyd_std','Hyd_max','Hyd_median']

def extract_charge(seq):
    aa_charge = {'R':1,'K':1,'H':0.1,'D':-1,'E':-1}
    acidic = ['D','E']
    basic = ['R','K','H']
    pi_dict = {'A':6.0,'R':10.76,'N':5.41,'D':2.77,'C':5.07,'Q':5.65,'E':3.22,'G':5.97,'H':7.59,'I':6.02,'L':5.98,'K':9.74,'M':5.74,'F':5.48,'P':6.35,'S':5.68,'T':5.60,'W':5.89,'Y':5.66,'V':5.96}
    total = len(seq)
    if total == 0:
        return [0,0,0,0], ['PI','Acidic_ratio','Basic_ratio','Net_charge']
    pi = np.mean([pi_dict[aa] for aa in seq if aa in pi_dict])
    acidic_ratio = sum(seq.count(aa) for aa in acidic)/total
    basic_ratio = sum(seq.count(aa) for aa in basic)/total
    net_charge = sum(aa_charge.get(aa,0) for aa in seq)
    return [pi, acidic_ratio, basic_ratio, net_charge], ['PI','Acidic_ratio','Basic_ratio','Net_charge']

def extract_sidechain(seq):
    side_vol = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
    polarizability = {'A':0.046,'R':0.291,'N':0.134,'D':0.105,'C':0.128,'Q':0.180,'E':0.151,'G':0.000,'H':0.230,'I':0.186,'L':0.186,'K':0.219,'M':0.214,'F':0.290,'P':0.131,'S':0.062,'T':0.108,'W':0.409,'Y':0.298,'V':0.140}
    total = len(seq)
    if total == 0:
        return [0]*5, ['Side_vol','Polarizability','H_donor','H_acceptor','Hyd_moment']
    vol_mean = np.mean([side_vol[aa] for aa in seq if aa in side_vol])
    polar_mean = np.mean([polarizability[aa] for aa in seq if aa in polarizability])
    h_donor = sum(seq.count(aa) for aa in ['R','K','H','N','Q'])/total
    h_acceptor = sum(seq.count(aa) for aa in ['D','E','N','Q'])/total
    kyte_doolittle = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    hyd_moment = np.mean([kyte_doolittle[aa]*side_vol[aa] for aa in seq if aa in kyte_doolittle and aa in side_vol])
    return [vol_mean, polar_mean, h_donor, h_acceptor, hyd_moment], ['Side_vol','Polarizability','H_donor','H_acceptor','Hyd_moment']

def extract_grouped(seq):
    polar = ['R','N','D','C','Q','E','H','S','T','W','Y']
    nonpolar = ['A','G','I','L','M','F','P','V']
    aromatic = ['F','W','Y']
    aliphatic = ['A','I','L','V']
    total = len(seq)
    if total == 0:
        return [0]*4, ['Polar_ratio','Nonpolar_ratio','Aromatic_ratio','Aliphatic_ratio']
    polar_ratio = sum(seq.count(aa) for aa in polar)/total
    nonpolar_ratio = sum(seq.count(aa) for aa in nonpolar)/total
    aromatic_ratio = sum(seq.count(aa) for aa in aromatic)/total
    aliphatic_ratio = sum(seq.count(aa) for aa in aliphatic)/total
    return [polar_ratio, nonpolar_ratio, aromatic_ratio, aliphatic_ratio], ['Polar_ratio','Nonpolar_ratio','Aromatic_ratio','Aliphatic_ratio']

def extract_sequence_features(seq):
    total = len(seq)
    if total == 0:
        return [0,0,0], ['Seq_len','Avg_MW','Charged_ratio']
    mw_dict = {'A':89.1,'R':174.2,'N':132.1,'D':133.1,'C':121.2,'Q':146.2,'E':147.1,'G':75.1,'H':155.2,'I':131.2,'L':131.2,'K':146.2,'M':149.2,'F':165.2,'P':115.1,'S':105.1,'T':119.1,'W':204.2,'Y':181.2,'V':117.1}
    avg_mw = np.mean([mw_dict[aa] for aa in seq if aa in mw_dict])
    charged = ['R','K','H','D','E']
    charged_ratio = sum(seq.count(aa) for aa in charged)/total
    return [total, avg_mw, charged_ratio], ['Seq_len','Avg_MW','Charged_ratio']

def extract_aaindex(seq):
    kyte_doolittle = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    pi_dict = {'A':6.0,'R':10.76,'N':5.41,'D':2.77,'C':5.07,'Q':5.65,'E':3.22,'G':5.97,'H':7.59,'I':6.02,'L':5.98,'K':9.74,'M':5.74,'F':5.48,'P':6.35,'S':5.68,'T':5.60,'W':5.89,'Y':5.66,'V':5.96}
    mw_dict = {'A':89.1,'R':174.2,'N':132.1,'D':133.1,'C':121.2,'Q':146.2,'E':147.1,'G':75.1,'H':155.2,'I':131.2,'L':131.2,'K':146.2,'M':149.2,'F':165.2,'P':115.1,'S':105.1,'T':119.1,'W':204.2,'Y':181.2,'V':117.1}
    surf_area = {'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
    total = len(seq)
    if total == 0:
        return [0]*8, ['Hyd_index','Avg_PI','Avg_MW','Avg_surf','Hyd_index2','Avg_PI2','Avg_MW3','Avg_surf2']
    hyd_index = np.mean([kyte_doolittle[aa] for aa in seq if aa in kyte_doolittle])
    avg_pi = np.mean([pi_dict[aa] for aa in seq if aa in pi_dict])
    avg_mw = np.mean([mw_dict[aa] for aa in seq if aa in mw_dict])
    avg_surf = np.mean([surf_area[aa] for aa in seq if aa in surf_area])
    return [hyd_index, avg_pi, avg_mw, avg_surf, hyd_index*2, avg_pi*2, avg_mw*2, avg_surf*2], ['Hyd_index','Avg_PI','Avg_MW','Avg_surf','Hyd_index2','Avg_PI2','Avg_MW3','Avg_surf2']

def extract_entropy(seq):
    aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    total = len(seq)
    aac = [seq.count(aa)/total if total>0 else 0 for aa in aa_list]
    aac_entropy = -sum(p*np.log2(p) if p>0 else 0 for p in aac)
    dpc_list = ['AL','AR','KR','LL','RR','AG','GD','VE','SN','TD','HI','YK','FM','PW','TS','EV','QA','NH']
    dpc = []
    for dp in dpc_list:
        count = 0
        for i in range(total-1):
            if seq[i]+seq[i+1] == dp:
                count +=1
        dpc.append(count/(total-1) if total>1 else 0)
    dpc_entropy = -sum(p*np.log2(p) if p>0 else 0 for p in dpc)
    return [aac_entropy, dpc_entropy], ['AAC_entropy','DPC_entropy']

def extract_all_features(fasta_path, out_path, dim=68):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    all_features = []
    all_ids = []
    feature_names = []
    for record in records:
        seq = str(record.seq).upper()
        seq = re.sub(r'\d+:', '', seq)
        seq = re.sub(r'[^A-Z]', '', seq)
        full_id = record.description
        clean_id = full_id
        all_ids.append(clean_id)
        aac, aac_names = extract_aac(seq)
        dpc, dpc_names = extract_dpc(seq)
        hyd, hyd_names = extract_hydrophobicity(seq)
        charge, charge_names = extract_charge(seq)
        sidechain, sidechain_names = extract_sidechain(seq)
        grouped, grouped_names = extract_grouped(seq)
        seq_feat, seq_names = extract_sequence_features(seq)
        aaindex, aaindex_names = extract_aaindex(seq)
        entropy, entropy_names = extract_entropy(seq)
        total_feat = aac + dpc + hyd + charge + sidechain + grouped + seq_feat + aaindex + entropy
        if len(total_feat) > dim:
            total_feat = total_feat[:dim]
        elif len(total_feat) < dim:
            total_feat += [0]*(dim - len(total_feat))
        all_features.append(total_feat)
        if not feature_names:
            feature_names = aac_names + dpc_names + hyd_names + charge_names + sidechain_names + grouped_names + seq_names + aaindex_names + entropy_names
            feature_names = feature_names[:dim] if len(feature_names) > dim else feature_names + [f"feat_{i}" for i in range(dim - len(feature_names))]
    df = pd.DataFrame(all_features, index=all_ids, columns=feature_names)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'protein_id'}, inplace=True)
    df.to_csv(out_path, index=False, encoding='gbk')

def load_data(file_path):
    df = pd.read_csv(file_path, encoding="gbk")
    id_col = "protein_id"
    label_col = "protein_label"
    feature_cols = [col for col in df.columns if col not in [id_col, label_col]]
    X = df[feature_cols].values
    y = df[label_col].values
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError(f"Label count is not 2: {unique_labels}")
    if not np.all(np.isin(y, [0, 1])):
        y = np.where(y == unique_labels[1], 1, 0)
    X = X.reshape(-1, len(feature_cols), 1, 1)
    y_onehot = to_categorical(y, num_classes=2)
    return X, y, y_onehot, feature_cols

def build_cnn_model(params, input_shape):
    model = Sequential()
    model.add(Conv2D(filters=params['filters'], kernel_size=(params['kernel_size'], 1), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['pool_size'], 1)))
    model.add(Conv2D(filters=params['filters']*2, kernel_size=(params['kernel_size'], 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['pool_size'], 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(2, activation='softmax'))
    optimizer = Adam(learning_rate=params['lr'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def hyperparameter_search(X_train, y_train_onehot, X_val, y_val_onehot, input_shape):
    param_grid = {
        'filters': [16, 32, 64],
        'kernel_size': [2, 3, 5],
        'pool_size': [2, 3],
        'dropout_rate': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0001]
    }
    best_params = None
    best_f1 = 0.0
    best_model = None
    search_results = []
    param_combinations = ParameterGrid(param_grid)
    for params in param_combinations:
        model = build_cnn_model(params, input_shape)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot), epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
        y_val_pred_prob = model.predict(X_val, verbose=0)
        y_val_pred = np.argmax(y_val_pred_prob, axis=1)
        val_f1 = f1_score(y_val, y_val_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        search_results.append({'params': params, 'val_f1': val_f1, 'val_acc': val_acc, 'val_loss': min(history.history['val_loss'])})
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params
            best_model = model
    with open(os.path.join(OUTPUT_DIR, "hyperparameter_search_results.json"), 'w', encoding='utf-8') as f:
        json.dump(search_results, f, ensure_ascii=False, indent=4)
    return best_model, best_params

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

def save_pred_prob(file_path, y_true, y_pred, y_pred_prob, save_name):
    df = pd.read_csv(file_path, encoding="gbk")
    df['true_label'] = y_true
    df['pred_label'] = y_pred
    df['adhesion_prob'] = y_pred_prob[:, 1]
    df['non_adhesion_prob'] = y_pred_prob[:, 0]
    df.to_csv(os.path.join(OUTPUT_DIR, save_name), index=False, encoding="gbk")

def run_training():
    train_path = os.path.join(CSV_DIR, "train_phys_68_standardized.csv")
    val_path = os.path.join(CSV_DIR, "val_phys_68_standardized.csv")
    X_train, y_train, y_train_onehot, feature_cols = load_data(train_path)
    X_val, y_val, y_val_onehot, _ = load_data(val_path)
    input_shape = (len(feature_cols), 1, 1)
    best_model, best_params = hyperparameter_search(X_train, y_train_onehot, X_val, y_val_onehot, input_shape)
    with open(os.path.join(OUTPUT_DIR, "best_hyperparameters.json"), 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=4)
    y_val_pred_prob = best_model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_prob)
    with open(os.path.join(OUTPUT_DIR, "val_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump({"val_metrics": val_metrics}, f, ensure_ascii=False, indent=4)
    save_pred_prob(val_path, y_val, y_val_pred, y_val_pred_prob, "val_predictions.csv")
    best_model.save(os.path.join(OUTPUT_DIR, "best_cnn_model.h5"))

if __name__ == "__main__":
    run_training()