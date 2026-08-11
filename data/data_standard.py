import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

input_dir = "./data/raw"
train_path = os.path.join(input_dir, "train_phys_68.csv")
val_path = os.path.join(input_dir, "val_phys_68.csv")
test_path = os.path.join(input_dir, "test_phys_68.csv")

output_dir = os.path.join(input_dir, "standardized")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_std_path = os.path.join(output_dir, "train_phys_68_standardized.csv")
val_std_path = os.path.join(output_dir, "val_phys_68_standardized.csv")
test_std_path = os.path.join(output_dir, "test_phys_68_standardized.csv")

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding="gbk")
        if "蛋白质ID" not in df.columns:
            df.rename(columns={df.columns[0]: "protein_id"}, inplace=True)
        else:
            df.rename(columns={"蛋白质ID": "protein_id"}, inplace=True)
        
        id_col = df["protein_id"].copy()
        feature_cols = [col for col in df.columns if col != "protein_id"]
        X = df[feature_cols].values
        
        return X, id_col, feature_cols
    except Exception as e:
        raise e

X_train, id_train, feature_cols = load_data(train_path)
X_val, id_val, _ = load_data(val_path)
X_test, id_test, _ = load_data(test_path)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

def save_standardized_data(X_std, id_col, feature_cols, save_path):
    df_std = pd.DataFrame(X_std, columns=feature_cols)
    df_std.insert(0, "protein_id", id_col.values)
    df_std.to_csv(save_path, index=False, encoding="gbk")

save_standardized_data(X_train_std, id_train, feature_cols, train_std_path)
save_standardized_data(X_val_std, id_val, feature_cols, val_std_path)
save_standardized_data(X_test_std, id_test, feature_cols, test_std_path)

scaler_save_path = os.path.join(output_dir, "standard_scaler.pkl")
joblib.dump(scaler, scaler_save_path)