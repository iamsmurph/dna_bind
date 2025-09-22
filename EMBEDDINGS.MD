# Embedding extraction scripts / chunks
/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/

## Scripts for extracting Boltz 2 embeddings by chunks
run_script_0.sh  run_script_1.sh run_script_2.sh  run_script_3.sh

# ARGUMENT and OUTPUT
boltz predict --devices 8 --cache /storage/ujp/.cache/huggingface --diffusion_samples 1 --write_full_pde --write_full_pae --write_contact_probs --write_tm_expected_value --write_cropped_embeddings ../../boltz_runs/uniprobe_subset_100tfs/chunk_0;

--write_cropped_embeddings
produces cropped_embeddings_*.pt
z ∈ ℝ^{N×N×128} (pair-token features)
indices ∈ ℕ^{N} (crop → full-length mapping)
s_full ∈ ℝ^{1×L×384} (full-length sequence stream snapshot)
⇒ covers “pair stream z” and gives a snapshot that can be down-sampled to act as s_inputs.
--write_contact_probs
produces contact_probs_*.npz   (contact_probs ∈ ℝ^{L×L})
⇒ optionally supplies the structure-based weight matrix used by Boltz-Lite (and can substitute for the “distogram bins” conditioning in the original head).
--write_full_pae (and --write_full_pde)
produce pae_*.npz, pde_*.npz   (pae/pde ∈ ℝ^{L×L})
⇒ alternative pair-error/confidence signals, convertible to contact-like weights.
--write_tm_expected_value
produces tm_expected_value_*.npz   (tm_expected_value ∈ ℝ^{L×L})
⇒ scalar structural confidence that Boltz-Lite stores as tm_mean.
(no explicit flag) → .cif
The predictor always writes a CIF/PDB (*_model_0.cif) with full atomic coordinates.
⇒ lets you reconstruct x_pred and build
token_to_rep_atom (map tokens → representative atom)
per-token masks (mol_type, affinity_token_mask, etc.)
true pairwise distances for the learned dist-bin embeddings (if you decide to replicate them).
(no explicit flag) → confidence_*_model_0.json
ipTM / pTM floats.
⇒ cached by Boltz-Lite as scalar features.

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
