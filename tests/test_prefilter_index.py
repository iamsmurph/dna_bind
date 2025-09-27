import os
import sys
import json
import glob
import tempfile
import importlib

import pytest


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip heavy prefilter test in CI")
def test_prefilter_index_small():
    """Smoke-test that prefilter index builds and marks all sampled dirs as valid.

    The test looks for prediction directories via the pattern given in the
    PREFILTER_TEST_GLOB environment variable.  If that env var is not set or
    yields no matches, the test is skipped.  Only the first 3 matched dirs are
    used so the heavy validation finishes quickly.
    """
    pred_glob = os.getenv(
        "PREFILTER_TEST_GLOB",
        "/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_0/predictions/*",
    )
    sample_dirs = sorted(glob.glob(pred_glob))[:3]
    if not sample_dirs:
        pytest.skip(f"No prediction dirs matched: {pred_glob}")

    # Build the index in a temporary directory
    with tempfile.TemporaryDirectory() as tmp_cache:
        # Build argument list for the CLI
        args = [
            "--prefilter-cache-refresh",
            "--pred_glob",
            pred_glob,
            "--cache-dir",
            tmp_cache,
        ]

        # The CLI's default labels_csv path is okay for this smoke test; if it
        # doesn't exist, the prefilter refresh still runs fine because labels
        # are not needed during validation.

        # Monkey-patch sys.argv and invoke the CLI programmatically so we avoid
        # spawning a subprocess (faster, easier to debug).
        from dna_bind_offline.train import cli as train_cli  # import locally

        old_argv = sys.argv
        try:
            sys.argv = ["cli.py"] + args
            train_cli.main()
        finally:
            sys.argv = old_argv

        index_path = os.path.join(tmp_cache, "prefilter_index.json")
        assert os.path.exists(index_path), "prefilter_index.json was not created"

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        # Basic sanity checks
        assert isinstance(index, dict) and "dirs" in index and "meta" in index
        assert index["meta"].get("dir_count") == len(sample_dirs)
        for d_info in index["dirs"].values():
            # All sampled dirs should be marked valid
            assert d_info.get("valid") is True
