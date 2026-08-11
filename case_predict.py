import pandas as pd
import numpy as np
import os
import re
import joblib
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Reshape, Multiply
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

predict_fasta_path = "./data/predict/external_test_set.txt"
model_path = "./models/triple_attention_model.h5"
scaler_68d_path = "./data/standardized/standard_scaler.pkl"
train_68d_feat_path = "./data/raw/train_phys_68.csv"
output_dir = "./output/predictions"
os.makedirs(output_dir, exist_ok=True)

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

def encode_amino_acids_for_6d(seq):
    seq = seq.upper()
    seq = seq.replace("G", "").replace("A", "").replace("P", "")
    seq = seq.translate(str.maketrans("VLIM", "VVVV"))
    seq = seq.translate(str.maketrans("STC", "SSS"))
    seq = seq.translate(str.maketrans("DE", "DD"))
    seq = seq.translate(str.maketrans("HKR", "HHH"))
    seq = seq.translate(str.maketrans("NQ", "NN"))
    seq = seq.translate(str.maketrans("FYW", "FFF"))
    return seq

def count_6d_frequency(seq):
    encoded_seq = encode_amino_acids_for_6d(seq)
    def count_duplets(sub_seq, duplet):
        count = 0
        if len(sub_seq) < 2:
            return 0.0
        for i in range(len(sub_seq)-1):
            if sub_seq[i] == duplet[0] and sub_seq[i+1] == duplet[1]:
                count += 1
        return round(count/(len(sub_seq)-1), 6) if len(sub_seq) > 1 else 0.0
    return [count_duplets(encoded_seq, "VV"), count_duplets(encoded_seq, "SS"),
            count_duplets(encoded_seq, "DD"), count_duplets(encoded_seq, "HH"),
            count_duplets(encoded_seq, "NN"), count_duplets(encoded_seq, "FF")]

def extract_aac(seq):
    aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    total = len(seq)
    return [seq.count(aa)/total if total > 0 else 0 for aa in aa_list]

def extract_dpc(seq):
    dpc_list = ['AL','AR','KR','LL','RR','AG','GD','VE','SN','TD','HI','YK','FM','PW','TS','EV','QA','NH']
    total = len(seq)
    dpc = []
    for dp in dpc_list:
        count = 0
        for i in range(total-1):
            if seq[i]+seq[i+1] == dp:
                count += 1
        dpc.append(count/(total-1) if total > 1 else 0)
    return dpc

def extract_hydrophobicity(seq):
    kyte_doolittle = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    hyd_values = [kyte_doolittle[aa] for aa in seq if aa in kyte_doolittle]
    if not hyd_values:
        return [0, 0, 0, 0]
    return [np.mean(hyd_values), np.std(hyd_values), np.max(hyd_values), np.median(hyd_values)]

def extract_charge(seq):
    aa_charge = {'R':1,'K':1,'H':0.1,'D':-1,'E':-1}
    acidic = ['D','E']
    basic = ['R','K','H']
    pi_dict = {'A':6.0,'R':10.76,'N':5.41,'D':2.77,'C':5.07,'Q':5.65,'E':3.22,'G':5.97,'H':7.59,'I':6.02,'L':5.98,'K':9.74,'M':5.74,'F':5.48,'P':6.35,'S':5.68,'T':5.60,'W':5.89,'Y':5.66,'V':5.96}
    total = len(seq)
    if total == 0:
        return [0, 0, 0, 0]
    pi = np.mean([pi_dict[aa] for aa in seq if aa in pi_dict])
    acidic_ratio = sum(seq.count(aa) for aa in acidic) / total
    basic_ratio = sum(seq.count(aa) for aa in basic) / total
    net_charge = sum(aa_charge.get(aa, 0) for aa in seq)
    return [pi, acidic_ratio, basic_ratio, net_charge]

def extract_sidechain(seq):
    side_vol = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
    polarizability = {'A':0.046,'R':0.291,'N':0.134,'D':0.105,'C':0.128,'Q':0.180,'E':0.151,'G':0.000,'H':0.230,'I':0.186,'L':0.186,'K':0.219,'M':0.214,'F':0.290,'P':0.131,'S':0.062,'T':0.108,'W':0.409,'Y':0.298,'V':0.140}
    total = len(seq)
    if total == 0:
        return [0]*5
    vol_mean = np.mean([side_vol[aa] for aa in seq if aa in side_vol])
    polar_mean = np.mean([polarizability[aa] for aa in seq if aa in polarizability])
    h_donor = sum(seq.count(aa) for aa in ['R','K','H','N','Q']) / total
    h_acceptor = sum(seq.count(aa) for aa in ['D','E','N','Q']) / total
    kyte_doolittle = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    hyd_moment = np.mean([kyte_doolittle[aa]*side_vol[aa] for aa in seq if aa in kyte_doolittle and aa in side_vol])
    return [vol_mean, polar_mean, h_donor, h_acceptor, hyd_moment]

def extract_grouped(seq):
    polar = ['R','N','D','C','Q','E','H','S','T','W','Y']
    nonpolar = ['A','G','I','L','M','F','P','V']
    aromatic = ['F','W','Y']
    aliphatic = ['A','I','L','V']
    total = len(seq)
    if total == 0:
        return [0]*4
    return [sum(seq.count(aa) for aa in polar)/total,
            sum(seq.count(aa) for aa in nonpolar)/total,
            sum(seq.count(aa) for aa in aromatic)/total,
            sum(seq.count(aa) for aa in aliphatic)/total]

def extract_sequence_features(seq):
    total = len(seq)
    if total == 0:
        return [0, 0, 0]
    mw_dict = {'A':89.1,'R':174.2,'N':132.1,'D':133.1,'C':121.2,'Q':146.2,'E':147.1,'G':75.1,'H':155.2,'I':131.2,'L':131.2,'K':146.2,'M':149.2,'F':165.2,'P':115.1,'S':105.1,'T':119.1,'W':204.2,'Y':181.2,'V':117.1}
    avg_mw = np.mean([mw_dict[aa] for aa in seq if aa in mw_dict])
    charged = ['R','K','H','D','E']
    charged_ratio = sum(seq.count(aa) for aa in charged) / total
    return [total, avg_mw, charged_ratio]

def extract_aaindex(seq):
    kyte_doolittle = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    pi_dict = {'A':6.0,'R':10.76,'N':5.41,'D':2.77,'C':5.07,'Q':5.65,'E':3.22,'G':5.97,'H':7.59,'I':6.02,'L':5.98,'K':9.74,'M':5.74,'F':5.48,'P':6.35,'S':5.68,'T':5.60,'W':5.89,'Y':5.66,'V':5.96}
    mw_dict = {'A':89.1,'R':174.2,'N':132.1,'D':133.1,'C':121.2,'Q':146.2,'E':147.1,'G':75.1,'H':155.2,'I':131.2,'L':131.2,'K':146.2,'M':149.2,'F':165.2,'P':115.1,'S':105.1,'T':119.1,'W':204.2,'Y':181.2,'V':117.1}
    surf_area = {'A':129,'R':274,'N':195,'D':193,'C':167,'Q':225,'E':223,'G':104,'H':224,'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
    total = len(seq)
    if total == 0:
        return [0]*8
    hyd = np.mean([kyte_doolittle[aa] for aa in seq if aa in kyte_doolittle])
    pi = np.mean([pi_dict[aa] for aa in seq if aa in pi_dict])
    mw = np.mean([mw_dict[aa] for aa in seq if aa in mw_dict])
    surf = np.mean([surf_area[aa] for aa in seq if aa in surf_area])
    return [hyd, pi, mw, surf, hyd*2, pi*2, mw*2, surf*2]

def extract_entropy(seq):
    aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    total = len(seq)
    aac = [seq.count(aa)/total if total > 0 else 0 for aa in aa_list]
    aac_entropy = -sum(p*np.log2(p) if p > 0 else 0 for p in aac)
    dpc_list = ['AL','AR','KR','LL','RR','AG','GD','VE','SN','TD','HI','YK','FM','PW','TS','EV','QA','NH']
    dpc = []
    for dp in dpc_list:
        count = 0
        for i in range(total-1):
            if seq[i]+seq[i+1] == dp:
                count += 1
        dpc.append(count/(total-1) if total > 1 else 0)
    dpc_entropy = -sum(p*np.log2(p) if p > 0 else 0 for p in dpc)
    return [aac_entropy, dpc_entropy]

def extract_68d_features(seq):
    aac = extract_aac(seq)
    dpc = extract_dpc(seq)
    hyd = extract_hydrophobicity(seq)
    charge = extract_charge(seq)
    sidechain = extract_sidechain(seq)
    grouped = extract_grouped(seq)
    seq_feat = extract_sequence_features(seq)
    aaindex = extract_aaindex(seq)
    entropy = extract_entropy(seq)
    feat = aac + dpc + hyd + charge + sidechain + grouped + seq_feat + aaindex + entropy
    return feat[:68] if len(feat) > 68 else feat + [0.0]*(68-len(feat))

def read_fasta_utf8_safe(filepath):
    sequences = []
    headers = []
    seq = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(''.join(seq))
                    seq = []
                headers.append(line[1:])
            else:
                clean_line = re.sub(r'[^A-Za-z]', '', line)
                seq.append(clean_line)
        if seq:
            sequences.append(''.join(seq))
    return headers, sequences

def main():
    df_train_68d = pd.read_csv(train_68d_feat_path, encoding='gbk')
    scaler_68d = joblib.load(scaler_68d_path)

    headers, sequences = read_fasta_utf8_safe(predict_fasta_path)

    all_6d = []
    all_68d = []
    ids = []

    for h, s in zip(headers, sequences):
        seq = s.upper()
        ids.append(h)
        f6 = count_6d_frequency(seq)
        f68 = extract_68d_features(seq)
        all_6d.append(f6)
        all_68d.append(f68)

    scaler_6d = StandardScaler()
    x6 = scaler_6d.fit_transform(all_6d)
    x68 = scaler_68d.transform(all_68d)
    X = np.concatenate([x6, x68], axis=1).reshape(-1, 74, 1, 1)

    with tf.keras.utils.custom_object_scope({
        "SelfAttentionLayer": SelfAttentionLayer,
        "CBAMSENetFusionLayer": CBAMSENetFusionLayer
    }):
        model = load_model(model_path, compile=False)

    prob = model.predict(X, verbose=0)
    pred = np.argmax(prob, axis=1)

    total = len(pred)
    normal = int(np.sum(pred == 0))
    adhesion = total - normal
    acc = round((normal + adhesion) / total, 4)

    df = pd.DataFrame({
        "protein_id": ids,
        "pred_label": pred,
        "pred_class": np.where(pred == 1, "adhesion", "normal"),
        "normal_prob": np.round(prob[:, 0], 4),
        "adhesion_prob": np.round(prob[:, 1], 4)
    })
    df.to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False, encoding='gbk')

    stat = pd.DataFrame({
        "predicted_normal": [normal],
        "predicted_adhesion": [adhesion],
        "prediction_accuracy": [acc]
    })
    stat.to_csv(os.path.join(output_dir, "prediction_stats.csv"), index=False, encoding='gbk')

if __name__ == "__main__":
    main()