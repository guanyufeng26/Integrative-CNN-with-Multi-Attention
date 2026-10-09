import os
import csv
from Bio import SeqIO
from itertools import product

ADHESIN_FASTA = r"data/raw/adhesin.fasta"
NON_ADHESIN_FASTA = r"data/raw/non_adhesin.fasta"
OUTPUT_ROOT = r"outputs/dipeptide_selection_result"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

aa_list = list("ARNDCEQGHILKMFPSTWYV")
all_400_dipep = ["".join(pair) for pair in product(aa_list, repeat=2)]

adhesin_seqs = []
non_adhesin_seqs = []

def read_fasta_seq_list(fasta_path):
    seq_list = []
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"File not found: {fasta_path}")
    print(f"Reading: {fasta_path}")
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq_raw = str(rec.seq).upper()
        clean_seq = "".join([ch for ch in seq_raw if ch in aa_list])
        if len(clean_seq) >= 2:
            seq_list.append(clean_seq)
    return seq_list

adhesin_seqs = read_fasta_seq_list(ADHESIN_FASTA)
non_adhesin_seqs = read_fasta_seq_list(NON_ADHESIN_FASTA)

print(f"\nNumber of adhesin sequences: {len(adhesin_seqs)}")
print(f"Number of non-adhesin sequences: {len(non_adhesin_seqs)}")

def calc_dipep_freq(seq_list, dipep_list):
    dipep_count = {dp:0 for dp in dipep_list}
    total_valid_pairs = 0
    for s in seq_list:
        n = len(s)
        for i in range(n-1):
            dp = s[i]+s[i+1]
            if dp in dipep_count:
                dipep_count[dp] += 1
            total_valid_pairs += 1
    freq_dict = {}
    if total_valid_pairs == 0:
        for dp in dipep_list:
            freq_dict[dp] = 0.0
    else:
        for dp in dipep_list:
            freq_dict[dp] = dipep_count[dp] / total_valid_pairs
    return freq_dict

freq_adhesin = calc_dipep_freq(adhesin_seqs, all_400_dipep)
freq_non = calc_dipep_freq(non_adhesin_seqs, all_400_dipep)

stat_rows = []
for dp in all_400_dipep:
    f_a = freq_adhesin[dp]
    f_n = freq_non[dp]
    diff_abs = abs(f_a - f_n)
    stat_rows.append({
        "dipeptide": dp,
        "freq_adhesin": f_a,
        "freq_non_adhesin": f_n,
        "abs_frequency_difference": diff_abs
    })

stat_rows_sorted = sorted(stat_rows, key=lambda x:x["abs_frequency_difference"], reverse=True)

csv_out = os.path.join(OUTPUT_ROOT, "all_400_dipeptide_stats.csv")
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["dipeptide","freq_adhesin","freq_non_adhesin","abs_frequency_difference"])
    writer.writeheader()
    writer.writerows(stat_rows_sorted)

top18 = [row["dipeptide"] for row in stat_rows_sorted[:18]]
top18_txt = os.path.join(OUTPUT_ROOT, "top18_dipeptides.txt")
with open(top18_txt, "w", encoding="utf-8") as f:
    f.write(",".join(top18)+"\n")
    for item in top18:
        f.write(item+"\n")

print("\nDipeptide selection completed.")
print(f"Full 400-dipeptide statistics file: {csv_out}")
print(f"Top-18 discriminative dipeptide list: {top18_txt}")
print(f"\nTop-18 dipeptides:\n{top18}")