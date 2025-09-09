#%%
import pickle as pkl
import os
import numpy as np
import argparse
#%%
def load_data(input_path):
    with open(input_path, "rb") as f:
        return pkl.load(f)

def preprocess(df):
    df = df[df["intensity"] >= 0]
    df = df.dropna(subset=["intensity", "uniprot", "nt"])
    df = df[df["mut"].isna()]
    df = df[~df["uniprot"].isin(['', 'n/a'])]
    return df

def transform_log1p(df):
    df = df.copy()
    df["intensity_log1p"] = np.log1p(df["intensity"].astype(float))
    df = df.drop(columns=["intensity"])  # remove raw intensity as requested
    return df

def dedup_log_then_mean(df):
    # Count duplicates for each uniprot-nt pair
    duplicate_counts = df.groupby(["uniprot", "nt"], dropna=False, observed=True).size()
    
    # Filter out pairs with more than 2 duplicates
    valid_pairs = duplicate_counts[duplicate_counts <= 2].index
    df_filtered = df.set_index(["uniprot", "nt"]).loc[valid_pairs].reset_index()
    
    mean_log = (
        df_filtered.groupby(["uniprot", "nt"], dropna=False, observed=True)["intensity_log1p"].mean().reset_index(name="intensity_log1p")
    )
    first_rows = df_filtered.sort_index().drop_duplicates(subset=["uniprot", "nt"], keep="first")
    dedup = first_rows.drop(columns=["intensity_log1p"]).merge(mean_log, on=["uniprot", "nt"], how="left")
    return dedup

def sample_per_tf(df, k, seed, min_per_tf=None):
    rng = np.random.default_rng(seed)
    df = df.copy()
    if min_per_tf is not None:
        counts = df.groupby("uniprot", observed=True).size()
        keep_uniprots = counts[counts >= min_per_tf].index
        df = df[df["uniprot"].isin(keep_uniprots)]
    df["_rand"] = rng.random(len(df))
    sampled = (
        df.sort_values(["uniprot", "_rand"]).groupby("uniprot", observed=True, group_keys=False).head(k)
    )
    return sampled.drop(columns=["_rand"]) 

def save_output(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Dedup (log->mean) and sample nt sequences per TF")
    parser.add_argument("--input", type=str, default="/data/rbg/users/seanmurphy/dna_bind/data/uniprobe_dataset.pkl", help="Path to input pickle dataset")
    parser.add_argument("--output", type=str, default="/data/rbg/users/seanmurphy/dna_bind/data/uniprobe_dedup_sampled.csv", help="Path to output CSV")
    parser.add_argument("--k", type=int, default=500, help="Number of nt sequences to sample per TF (default: 400)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampling (default: 1337)")
    return parser.parse_args()

def main():
    args = parse_args()
    data = load_data(args.input)
    data = preprocess(data)
    data = transform_log1p(data)
    data_dedup = dedup_log_then_mean(data)
    sampled = sample_per_tf(data_dedup, args.k, args.seed, min_per_tf=args.k)
    save_output(sampled, args.output)

if __name__ == "__main__":
    main()
