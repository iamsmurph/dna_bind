# %%

import pandas as pd
import sqlite3
from tqdm import tqdm

from pathlib import Path
import logging
import pickle as pkl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# SHA256 hashes of the tarballs in the data/pbm/uniprobe/ folder
# these were downloaded from the UniPROBE website: https://uniprobe.org/downloads/

# 5e94e3ee8b5f00d29eb5386e35431e99f0cb5c9e46203ae59421a64957543750  BAR15A_PBM_full.tgz
# c9767cfcf5560f844d8e31ac1615244d08073e2b323da5d6efb8845ad91e0c95  CB11_PBM_full.tgz
# ac85ce866594327c0fa96bf17cc1152dff63b9ded2043d1df3875350fa3cb865  Cell08_PBM_full.tgz
# eee90483d8ecfe13bda0e2117a4843b13fcebc444b0110c4cf5af3a9845a0ef3  Cell09_PBM_full.tgz
# e8b4c07ba05cc0dfa44e9304a29dd1a7c52dd6b4b7d6b3ce0274951d797276ca  CR09_PBM_full.tgz
# 5992b747aecec68bde7ef9024552033aa09dc9157c1839e8ff027d85bb9f7254  DEV12_PBM_full.tgz
# 026d2f793f54a1b4429aeae5b091c612ecf039ae262817cb0392c02d6a650bd6  EMBO10_PBM_full.tgz
# ba4dffb8f90a9e14ec7a8a641781080849e27df95731f3504b205c32e3f33d24  GB11_PBM_full.tgz
# 0504b59254de7a3b5a9a056a29d5e6718423eabe7bd5362ad6ede0a5c49b2f17  GD09_PBM_full.tgz
# c7b7e432c7690fd2c7fc455935e43158cd86945f3aa1b4b4d2fa4dd28950f646  GD12_PBM_full.tgz
# 9c80fa523aad0a176385c2870cb1d0efc347606b1299cc8c35063d2368645f3b  GD13_PBM_full.tgz
# 71e670410b1b6fc2da99df6f404e2b16a005934cb2dfb32088fea0d32b8be452  GR09_PBM_full.tgz
# e63385a5866c34e452dc6eb9392f493518f1b0b80f2e4c02a2ed7ed8d9748ca0  KUR17A_PBM_full.tgz
# 967a4133f69025d4530bd9afd8691363d2584a0be13d60b45380d773395458f0  LIN14B_PBM_full.tgz
# 2cc96408ccf89c4f44b5db7e8ba2a688444001714f35e70c946d2171398e0be9  LIU18A_PBM_full.tgz
# 299b9aa833f970a18e45a8302f7430dcdd6ab522c71688527f1314a1d42b9c08  LIU18B_PBM_full.tgz
# e45cdd5518af9bd41f8db5114110bc682fd55cd74c29d02d62d52f64b8e7bf01  MAR17A_PBM_full.tgz
# 80d6d620f1bcdb8a1e880dab61a0c5c8f5d57fb6cadd4d9c0f6b82b6a24e5b97  MBE14_PBM_full.tgz
# e471ff9c36ead024f5bdaac10491600f13bbc0abcbdef248844fd0a43098bb4b  NAR10_PBM_full.tgz
# 5368b41d005de8a13a06f5459adf277a5948c384f908d492038aed5ea4a3115a  NAR11_PBM_full.tgz
# 8c74d6bb96f3bfa490e6d3398a8c6aa23ad63320520c1d4f764cfde0e7723309  NBT06_PBM_full.tgz
# a0fff7fdba493bc709c5cb2770ba7bd86ddf5e8ed4ba7eeadf27c333e1d2fd30  Path10_PBM_full.tgz
# 24e298200768665076e4303c33e51c9ab0df8c6899e36759b6d47e39ecd39c37  PNAS12_PBM_full.tgz
# 0828d2aa3475b24e919b727092e9a12ebbb63b43ff95d6b311a5ca4386a2b245  PNAS13_PBM_full.tgz
# 6942512e2c9307d1e99465036d0acf30648a555f283f5b21634c2d9cd441e877  PO10_PBM_full.tgz
# 657f4b22d6c40631cde95dfe2d80b92e70da4a0b15f25e5e05915d6c6ba302eb  PP15_PBM_full.tgz
# c35ffa21f4d5df850799f8180e6f63917be7c54fe88c24e2390ec8e8b41b7d1f  RAD13A_PBM_full.tgz
# f695d69dcf7d9cbc9dd444d85a1a6a716d5a113e0da90012bb970153c2f5448c  ROG18A_PBM_full.tgz
# 8c43b4782f82bbcbf4e422dd664770913ae084d817a7729f7bb65c886016162d  SAN17A_PBM_full.tgz
# afbf50dbdb7eabd724b37f67752f3a910d5b9e5309ecd17241bdca06eeb575b7  SCI09_PBM_full.tgz
# d2aecd6309405da00b1200c74e5cdea70911a94b19870ade8d561d511bf2fec7  SHO18A_PBM_full.tgz


# SHA256 Hash of the SQL zip file:
# 83db07bf75a194922ce5ac682671cb65726194d11146c69ddd988d74d0c76af8  SQL.zip


# %%

# Read the UniPROBE database
conn = sqlite3.connect("/data/rbg/users/ujp/dnabind/data/pbm/uniprobe/uniprobe.db")
tables = {}
names = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)[
    "name"
].tolist()
subset_names = ["publication_ids", "gene_ids", "genomic_info"]
assert set(subset_names).issubset(
    set(names)
), "Subset names are not a subset of all names"
logger.info(f"Available tables: {names}")

# Read all data from the database (adjust table name as needed)
subset_names = names

for name in tqdm(subset_names):
    tables[name] = pd.read_sql_query(f"SELECT * FROM {name};", conn)
conn.close()

# %%

tables.keys()
# %%
# Show columns for each table
for table_name, df in tables.items():
    print(f"Table '{table_name}' columns: {list(df.columns)}")
    print(f"Table '{table_name}' shape: {df.shape}")
    print("---")


# %%
tables["clone_to_protseq"]

# %%

# %%


pubnames = set(tables["publication_ids"]["folder_name"].unique())
individual = set([g.name for g in Path("../../data/pbm/uniprobe/").glob("*")])
expected_missing = {"MMB08", "PNAS08"}
if pubnames - individual - expected_missing:
    logger.warning(f"Missing publications: {pubnames - individual}. ")

# %%

# deleted GR09/trash/ subfolder, looks like a trash folder

data_dir = Path("../../data/pbm/uniprobe/")
data_files = list(data_dir.glob("**/*combinatorial*.txt")) + list(
    data_dir.glob("**/*deBruijn*.txt")
)
data_files = sorted(set(data_files) - set(data_dir.glob("downloads/**/*")))

# %%

merged = tables["gene_ids"].merge(
    tables["publication_ids"], on="publication_id", how="left"
)
merged = merged.merge(tables["genomic_info"], on=["gene_name", "species"], how="left")
merged["gene_id_name"] = merged["gene_id_name"].apply(lambda x: x.upper())

# %%
tables
# %%


num_cols = []
pubs_found = set()
frames = []
data_file_relpaths = []
for i, data_file in tqdm(enumerate(data_files), total=len(data_files)):
    data_file = data_file.relative_to(data_dir)
    data_file_relpaths.append(str(data_file))
    pub = data_file.parts[0]

    pubs_found.add(pub)
    gene = data_file.parts[1].upper()
    species = None

    mut = None
    replicate = None
    version = None
    probeset = "combinatorial" if "combinatorial" in data_file.name else "deBruijn"

    if len(data_file.parts) == 3:
        if "v1.txt" in data_file.name or "_v1_" in data_file.name:
            version = "v1"
        elif "v2.txt" in data_file.name or "_v2_" in data_file.name:
            version = "v2"
    elif len(data_file.parts) == 4:
        version = data_file.parts[2]
        pass
    elif len(data_file.parts) == 5:
        mut = data_file.parts[2]
        replicate = data_file.parts[3]

        pass

    meta_info = merged.loc[
        (merged["folder_name"] == pub) & (merged["gene_id_name"] == gene)
    ]
    if len(meta_info) != 1:
        logger.warning(f"Multiple uniprot entries for {pub}/{gene}. Skipping...")
        continue
    uniprot = meta_info["uniprot"].values[0]
    species = meta_info["species"].values[0]

    # Read first row to check for headers
    with open(data_dir / data_file, "r") as f:
        first_row = f.readline().strip().split("\t")

    # Check if any column contains 'Sequence'
    has_sequence_header = any("Sequence" in col for col in first_row)
    if has_sequence_header:
        df = pd.read_csv(
            data_dir / data_file,
            sep="\t",
            names=["nt", "intensity"],
            skiprows=1,
        )
    else:
        df = pd.read_csv(
            data_dir / data_file,
            sep="\t",
            names=["intensity", "nt"],
        )

    df["pub"] = pub
    df["data_file_relpath"] = i
    df["gene_name"] = gene
    df["uniprot"] = uniprot
    df["species"] = species
    df["mut"] = mut
    df["replicate"] = replicate
    df["version"] = version
    df["nt"] = df["nt"].apply(lambda x: x.strip().upper() if x.strip() else None)
    df["probeset"] = probeset
    if df["intensity"].dtype == "object":
        df["intensity"] = (
            df["intensity"].apply(lambda x: x if x.strip() else None).astype(float)
        )
    df = df[
        [
            "pub",
            "gene_name",
            "uniprot",
            "species",
            "mut",
            "replicate",
            "version",
            "probeset",
            "data_file_relpath",
            "nt",
            "intensity",
        ]
    ]
    frames.append(df)


# %%
data = pd.concat(frames)
# %%
data = data.dropna(subset=["uniprot", "nt", "intensity"])
# %%

# Convert categorical columns to categorical dtype
categorical_cols = [
    "pub",
    "gene_name",
    "uniprot",
    "species",
    "mut",
    "replicate",
    "version",
    "probeset",
]
for col in categorical_cols:
    data[col] = data[col].astype("category")

data["data_file_relpath"] = pd.Categorical.from_codes(
    data["data_file_relpath"], data_file_relpaths
)

with open(data_dir / "uniprobe_probeset_dataset.pkl", "wb") as f:
    pkl.dump(data, f)
# %%
# %%
