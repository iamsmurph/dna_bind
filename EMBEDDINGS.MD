# Embedding extraction scripts / chunks
/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/

## Scripts for extracting Boltz 2 embeddings by chunks
run_script_0.sh  run_script_1.sh run_script_2.sh  run_script_3.sh

# ARGUMENT and OUTPUT
boltz predict --devices 8 --cache /storage/ujp/.cache/huggingface --diffusion_samples 1 --write_full_pde --write_full_pae --write_contact_probs --write_tm_expected_value --write_cropped_embeddings ../../boltz_runs/uniprobe_subset_100tfs/chunk_0;

## Chunk Folders
boltz_results_chunk_0  boltz_results_chunk_1  boltz_results_chunk_2  boltz_results_chunk_3
## Example .pt file
ls /data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_0/predictions/P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG/

confidence_P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG_model_0.json
contact_probs_P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG.npz
cropped_embeddings_P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG.pt
P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG_model_0.cif
pae_P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG_model_0.npz
pde_P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG_model_0.npz
plddt_P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG_model_0.npz
tm_expected_value_P38128_9.964522083305246_AATTTATATTTCCCCGCATATCCGGCTCACTCCCGGGTCTGTGTTCCGTTGTCCGTGCTG_model_0.npz


## s_full vs s_inputs (and what's stored in cropped .pt)

- **s_full**: full-length single-token representations from the trunk for all tokens (protein + DNA). Stored in cropped `.pt` as `s_full`.
- **s_inputs**: the single-token representations aligned to the cropped tokens used with `z` inside the affinity module. Relationship: **s_inputs = s_full[indices]**, where `indices` are the crop→full token indices saved in the `.pt`.
- **What cropped `.pt` files contain**: `z` (or `z_crop`) with shape `[Lc, Lc, token_z]`, `indices` of length `Lc`, and `s_full` (sometimes `s_crop` in some runs). They do not store `s_inputs` directly.

Code references:

```569:577:/data/rbg/users/seanmurphy/dna_bind/boltz/src/boltz/model/models/boltz2.py
dict_out["cropped_embeddings"] = {}
for b in range(z.shape[0]):
    dict_out["cropped_embeddings"][b] = {}
    dict_out["cropped_embeddings"][b]["indices"] = selected_indices[b]
    dict_out["cropped_embeddings"][b]["z"] = z[b][selected_indices[b]][:, selected_indices[b]]
    dict_out["cropped_embeddings"][b]["s_full"] = s
```

```86:94:/data/rbg/users/seanmurphy/dna_bind/boltz/src/boltz/model/modules/affinity.py
z = self.z_linear(self.z_norm(z))
z = z.repeat_interleave(multiplicity, 0)

z = (
    z
    + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
    + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
)
```

In the TF–DNA training utilities, `s_inputs` is reconstructed as a cropped view of `s_full` (or uses `s_crop` when present):

```187:213:/data/rbg/users/seanmurphy/dna_bind/scripts/tfdna_affinity.py
if "s_crop" in inner:
    s_proxy = inner["s_crop"]
    ...
    if s_proxy.shape[0] != Lc:
        raise ValueError("s_crop not aligned to crop")
    c_single = s_proxy.shape[-1]
elif "s_full" in inner:
    s_full = inner["s_full"]
    ...
    max_needed = int(crop_to_full.max()) + 1
    if s_full.shape[0] < max_needed:
        raise AssertionError("s_full length < max crop_to_full index + 1")
    s_proxy = s_full[crop_to_full]
    c_single = s_proxy.shape[-1]
```
