import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
plt.switch_backend('Agg')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

COLOR_NEG = '#c2dfc0'
COLOR_POS = '#f1beb2'
colors = [COLOR_NEG, COLOR_POS]
markers = ['o', 'o']
label_names = ['Negative', 'Positive']

save_dir = "./output/umap"
os.makedirs(save_dir, exist_ok=True)

path_6d = "./data/raw/test_6.csv"
path_68d = "./data/standardized/test_phys_68_standardized.csv"
df6 = pd.read_csv(path_6d, encoding='utf-8')
df68 = pd.read_csv(path_68d, encoding='gbk')

feats6 = ['VV_freq', 'SS_freq', 'DD_freq', 'HH_freq', 'NN_freq', 'FF_freq']
X6 = df6[feats6].values

feats68 = [c for c in df68.columns if c not in ["protein_id", "protein_label"]]
X68 = df68[feats68].values

X_raw = np.concatenate([X6, X68], axis=1)
y_raw = df68['protein_label'].values

data_dir = "./data/features"
feat_control = np.load(f"{data_dir}/control_model_test_features_last_layer.npy")
feat_best = np.load(f"{data_dir}/best_model_test_features_last_layer.npy")
labels = np.load(f"{data_dir}/test_labels.npy")

def get_test_data():
    from sklearn.preprocessing import StandardScaler
    df6 = pd.read_csv("./data/raw/test_6.csv", encoding='utf-8')
    df68 = pd.read_csv("./data/standardized/test_phys_68_standardized.csv", encoding='gbk')
    x6 = StandardScaler().fit_transform(df6[['VV_freq', 'SS_freq', 'DD_freq', 'HH_freq', 'NN_freq', 'FF_freq']].values)
    x68 = df68[[c for c in df68.columns if c not in ["protein_id", "protein_label"]]].values
    return np.concatenate([x6, x68], axis=1).reshape(-1, 74, 1, 1), df68['protein_label'].values

X_test, y_test = get_test_data()

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Reshape, Dense, Multiply, GlobalMaxPooling2D, Concatenate

class SEBlock(Layer):
    def __init__(self, ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, s):
        self.ch = s[-1]
        self.p = GlobalAveragePooling2D()
        self.r = Reshape((1, 1, self.ch))
        self.f1 = Dense(self.ch // self.ratio, 'relu')
        self.f2 = Dense(self.ch, 'sigmoid')
        self.m = Multiply()

    def call(self, x):
        o = self.p(x)
        o = self.r(o)
        o = self.f1(o)
        o = self.f2(o)
        return self.m([x, o])

class CBAMSENetFusionLayer(Layer):
    def __init__(self, reduction_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.r = reduction_ratio

    def build(self, s):
        self.c = s[-1]
        self.d = self.c // self.r
        self.ca = GlobalAveragePooling2D()
        self.cm = GlobalMaxPooling2D()
        self.c1 = Dense(self.d, 'relu')
        self.c2 = Dense(self.c, 'sigmoid')
        self.sa = GlobalAveragePooling2D()
        self.s1 = Dense(self.d, 'relu')
        self.s2 = Dense(self.c, 'sigmoid')

    def call(self, i):
        ca = self.ca(i)
        cm = self.cm(i)
        c = Concatenate()([ca, cm])
        c = self.c1(c)
        c = self.c2(c)
        c = Reshape((1, 1, self.c))(c)
        co = Multiply()([i, c])
        s = self.sa(co)
        s = self.s1(s)
        s = self.s2(s)
        s = Reshape((1, 1, self.c))(s)
        return Multiply()([co, s])

model_se = load_model("./models/CNN+SENet_best_model.h5", custom_objects={"SEBlock": SEBlock})
model_cbam = load_model("./models/CNN_CBAM_SENet_fusion_model.h5",
                        custom_objects={"CBAMSENetFusionLayer": CBAMSENetFusionLayer})
f_se = tf.keras.Model(model_se.input, model_se.layers[-2].output).predict(X_test, verbose=0)
f_cbam = tf.keras.Model(model_cbam.input, model_cbam.layers[-2].output).predict(X_test, verbose=0)

umap_raw = UMAP(n_components=2, n_neighbors=5, min_dist=1, random_state=42).fit_transform(X_raw)
umap_ctrl = UMAP(random_state=42).fit_transform(feat_control)
umap_senet = UMAP(random_state=42).fit_transform(f_se)
umap_cbam = UMAP(random_state=42).fit_transform(f_cbam)
umap_best = UMAP(random_state=42).fit_transform(feat_best)

all_umaps = [
    (umap_raw, y_raw, "74D Feature"),
    (umap_ctrl, labels, "74D CNN"),
    (umap_senet, y_test, "CNN+SENet"),
    (umap_cbam, y_test, "CNN+SENet+CBAM"),
    (umap_best, labels, "CNN+SENet+CBAM+global self-attention")
]

fig = plt.figure(figsize=(24, 14))

w = 0.28
h = 0.38
y_top = 0.55
gap_abc = 0.04

ax_a = fig.add_axes([0.05, y_top, w, h])
ax_b = fig.add_axes([0.05 + w + gap_abc, y_top, w, h])
ax_c = fig.add_axes([0.05 + w*2 + gap_abc*2, y_top, w, h])

gap_de = 0.12
center = 0.5
d_x = center - w - gap_de/2
e_x = center + gap_de/2
ax_d = fig.add_axes([d_x, 0.08, w, h])
ax_e = fig.add_axes([e_x, 0.08, w, h])

axes = [ax_a, ax_b, ax_c, ax_d, ax_e]
mark_labels = ['a', 'b', 'c', 'd', 'e']

for idx, (ax, (emb, y, title)) in enumerate(zip(axes, all_umaps)):
    for i in [0, 1]:
        m = y == i
        ax.scatter(emb[m, 0], emb[m, 1], c=colors[i], marker=markers[i], s=50, alpha=0.8, label=label_names[i])

    ax.set_title(title, fontsize=22, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP 1', fontsize=18)
    ax.set_ylabel('UMAP 2', fontsize=18)
    ax.grid(alpha=0.3, zorder=0)
    ax.text(-0.08, 1.08, mark_labels[idx], transform=ax.transAxes, fontsize=28, fontweight='bold')

    handles, labels_legend = ax.get_legend_handles_labels()
    ax.legend(
        handles=[handles[1], handles[0]],
        labels=[labels_legend[1], labels_legend[0]],
        loc='upper right', fontsize=14, framealpha=0.6
    )

out = os.path.join(save_dir, "ALL_UMAP_5in1.png")
plt.savefig(out, dpi=600, bbox_inches='tight')
plt.close()