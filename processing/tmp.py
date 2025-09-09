#%%
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import numpy as np
import argparse
#%%
df = pd.read_csv("/data/rbg/users/seanmurphy/dna_bind/data/uniprobe_dedup_sampled.csv")
#%%
# Random sample 200 unique uniprot values
unique_uniprots = df['uniprot'].unique()
rng = np.random.default_rng(42)  # Using a fixed seed for reproducibility
sampled_uniprots = rng.choice(unique_uniprots, size=min(100, len(unique_uniprots)), replace=False)

# Subset df by those values
df = df[df['uniprot'].isin(sampled_uniprots)]


# %%
# Save the subsetted dataframe
output_path = "/data/rbg/users/seanmurphy/dna_bind/data/uniprobe_subset_100tfs.csv"
df.to_csv(output_path, index=False)
print(f"Saved subsetted dataframe with {len(df)} rows and {len(df['uniprot'].unique())} unique TFs to {output_path}")

# %%


# %%
# Plot histograms of intensity_log1p values for each unique uniprot
unique_uniprots = df['uniprot'].unique()

# Set up the plot layout
n_plots = len(unique_uniprots)
n_cols = 4  # Number of columns in the subplot grid
n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
axes = axes.flatten() if n_plots > 1 else [axes]  # Ensure axes is always a list

for i, uniprot in enumerate(unique_uniprots):
    # Get data for this uniprot
    uniprot_data = df[df['uniprot'] == uniprot]['intensity_log1p']
    
    # Plot histogram
    axes[i].hist(uniprot_data, bins=20, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{uniprot}\n(n={len(uniprot_data)})', fontsize=10)
    axes[i].set_xlabel('intensity_log1p')
    axes[i].set_ylabel('Frequency')

# Hide any unused subplots
for i in range(n_plots, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# %%
plt.hist(df['intensity_log1p'], bins=20, alpha=0.7, edgecolor='black')

# %%
