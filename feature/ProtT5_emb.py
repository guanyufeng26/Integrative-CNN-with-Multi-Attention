import os
import re
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import ParameterGrid

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATA_DIR = "./data/raw"
OUTPUT_DIR = "./output"
MODEL_DIR = "./prot_bert_model"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_6.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_6.csv")

MAX_LEN = 512
GAP_CHAR = "-"
TRUNCATE_SIDE = "N"
PAD_SIDE = "C"
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

PROTBERT_PATH = MODEL_DIR
BATCH_SIZE = 16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FEAT_DTYPE = np.float16

PCA_DIM_LIST = [128, 256, 512]
INCLUDE_BASELINE = True
SEED = 42
EPOCHS = 80
BATCH_SIZE_CNN = 64
PATIENCE = 12
LEARNING_RATE = 5e-4
PRED_THRESHOLD = 0.5
PARAM_GRID = {
    "conv1_filters": [32, 64],
    "conv1_kernel": [3, 5],
    "pool_kernel": [2, 4],
    "dropout_prob": [0.3, 0.4],
    "conv2_filters": [32, 64],
    "l2_reg": [1e-4]
}

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def read_raw_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    protein_ids = df.iloc[:, 0].tolist()
    raw_seqs = df.iloc[:, 1].tolist()
    labels = df.iloc[:, 2].tolist()
    clean_seqs = []
    for seq in raw_seqs:
        clean_seq = "".join([c for c in str(seq).strip() if c in VALID_AMINO_ACIDS])
        clean_seqs.append(clean_seq)
    return df, protein_ids, clean_seqs, labels

def process_sequence(seq):
    seq_len = len(seq)
    if seq_len > MAX_LEN:
        if TRUNCATE_SIDE == "N":
            processed = seq[:MAX_LEN]
        else:
            processed = seq[-MAX_LEN:]
    elif seq_len < MAX_LEN:
        pad_length = MAX_LEN - seq_len
        if PAD_SIDE == "C":
            processed = seq + GAP_CHAR * pad_length
        else:
            processed = GAP_CHAR * pad_length + seq
    else:
        processed = seq
    return processed

def load_protbert():
    tokenizer = BertTokenizer.from_pretrained(PROTBERT_PATH, do_lower_case=False)
    if "-" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["-"])
    model = BertModel.from_pretrained(PROTBERT_PATH)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(DEVICE).eval()
    return tokenizer, model, model.config.hidden_size

def extract_features(tokenizer, model, seqs, hidden_size):
    all_feats = []
    total_batches = len(seqs) // BATCH_SIZE + (1 if len(seqs) % BATCH_SIZE != 0 else 0)
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, len(seqs))
        batch_seqs = seqs[start:end]
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding="max_length",
            truncation=False,
            max_length=MAX_LEN + 2,
            add_special_tokens=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size())
            mask = mask[:, 1:-1, :]
            hidden = hidden[:, 1:-1, :]
            sum_emb = torch.sum(hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            batch_feats = (sum_emb / sum_mask).cpu().numpy().astype(FEAT_DTYPE)
        all_feats.extend(batch_feats)
    return np.vstack(all_feats)

class ProteinDataset(Dataset):
    def __init__(self, features, labels, target_dim, scaler=None, pca_model=None, is_train=False):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.target_dim = target_dim
        if is_train:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            if self.target_dim != 1024:
                self.pca_model = PCA(n_components=self.target_dim, random_state=SEED)
                self.features = self.pca_model.fit_transform(self.features)
            else:
                self.pca_model = None
        else:
            self.features = scaler.transform(self.features)
            if self.target_dim != 1024:
                self.features = pca_model.transform(self.features)
        self.features = self.features.reshape(-1, 1, self.features.shape[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx])

class ProteinCNN(nn.Module):
    def __init__(self, params, input_dim):
        super(ProteinCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, params["conv1_filters"], params["conv1_kernel"], padding="same", bias=False)
        self.bn1 = nn.BatchNorm1d(params["conv1_filters"])
        self.pool1 = nn.MaxPool1d(params["pool_kernel"], stride=params["pool_kernel"])
        self.conv2 = nn.Conv1d(params["conv1_filters"], params["conv2_filters"], 3, padding="same", bias=False)
        self.bn2 = nn.BatchNorm1d(params["conv2_filters"])
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.dropout = nn.Dropout(params["dropout_prob"])
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_dim)
            dummy = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool2(F.relu(self.bn2(self.conv2(dummy))))
            fc_input = dummy.numel()
        self.fc1 = nn.Linear(fc_input, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x)).squeeze()

def train_model(model, train_loader, val_loader, params):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=params["l2_reg"])
    criterion = nn.BCELoss()
    best_loss = float("inf")
    best_state = None
    patience = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE).float()
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE).float()
                loss = criterion(model(feats), labels)
                val_loss += loss.item() * feats.size(0)
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model

def evaluate_model(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(DEVICE)
            probs = model(feats).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= PRED_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    sn = tp / (tp + fn) if tp+fn>0 else 0.0
    sp = tn / (tn + fp) if tn+fp>0 else 0.0
    auc_roc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels))>1 else 0.0
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = auc(rec, prec)
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "ACC": round(acc,4), "MCC": round(mcc,4), "F1": round(f1,4),
            "AUC": round(auc_roc,4), "AUPR": round(auc_pr,4),
            "Sn": round(sn,4), "Sp": round(sp,4)}

def run_pca_experiment(train_feats, train_labels, val_feats, val_labels):
    test_dims = PCA_DIM_LIST[:]
    if INCLUDE_BASELINE:
        test_dims.append(1024)
    all_results = []
    for pca_dim in test_dims:
        train_ds = ProteinDataset(train_feats, train_labels, pca_dim, is_train=True)
        scaler, pca_model = train_ds.scaler, train_ds.pca_model
        val_ds = ProteinDataset(val_feats, val_labels, pca_dim, scaler=scaler, pca_model=pca_model)
        class_weights = 1.0 / np.bincount(train_labels)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, BATCH_SIZE_CNN, sampler=sampler, pin_memory=DEVICE.type=="cuda")
        val_loader = DataLoader(val_ds, BATCH_SIZE_CNN, shuffle=False, pin_memory=DEVICE.type=="cuda")
        best_mcc = -1.0
        best_params = None
        best_model = None
        for params in ParameterGrid(PARAM_GRID):
            model = ProteinCNN(params, input_dim=pca_dim).to(DEVICE)
            model = train_model(model, train_loader, val_loader, params)
            val_metrics = evaluate_model(model, val_loader)
            if val_metrics["MCC"] > best_mcc:
                best_mcc = val_metrics["MCC"]
                best_params = params
                best_model = model
        val_metrics_final = evaluate_model(best_model, val_loader)
        result = {"PCA_dim": pca_dim, "is_baseline": (pca_dim==1024),
                  "best_params": best_params, "val_metrics": val_metrics_final}
        all_results.append(result)
    return all_results

def main():
    train_df, train_ids, train_raw, train_labels = read_raw_csv(TRAIN_CSV)
    val_df, val_ids, val_raw, val_labels = read_raw_csv(VAL_CSV)

    train_seqs = [process_sequence(s) for s in train_raw]
    val_seqs = [process_sequence(s) for s in val_raw]

    tokenizer, model, hidden_size = load_protbert()
    train_feats = extract_features(tokenizer, model, train_seqs, hidden_size)
    val_feats = extract_features(tokenizer, model, val_seqs, hidden_size)

    results = run_pca_experiment(train_feats, np.array(train_labels),
                                 val_feats, np.array(val_labels))

    compare_df = pd.DataFrame([{
        "PCA_dim": r["PCA_dim"],
        "Baseline": r["is_baseline"],
        "ACC": r["val_metrics"]["ACC"],
        "MCC": r["val_metrics"]["MCC"],
        "F1": r["val_metrics"]["F1"],
        "AUC": r["val_metrics"]["AUC"],
        "AUPR": r["val_metrics"]["AUPR"],
        "Sn": r["val_metrics"]["Sn"],
        "Sp": r["val_metrics"]["Sp"]
    } for r in results])
    compare_df.to_csv(os.path.join(OUTPUT_DIR, "pca_comparison.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUTPUT_DIR, "pca_full_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()