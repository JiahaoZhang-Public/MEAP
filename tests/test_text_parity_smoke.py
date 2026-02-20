import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    os.environ.get("MM_EAP_IG_RUN_PARITY_SMOKE") != "1",
    reason="Set MM_EAP_IG_RUN_PARITY_SMOKE=1 to run parity smoke integration.",
)
def test_gpt2_eap_ig_activations_parity_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "gpt2_parity_smoke.json"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "test_text_vendor_parity.py"),
        "--models",
        "gpt2-small",
        "--methods",
        "EAP-IG-activations",
        "--n-samples",
        "1",
        "--batch-size",
        "1",
        "--ig-steps",
        "1",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--strict",
        "--quiet",
        "--output",
        str(report_path),
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload["results"]
    assert len(rows) == 1
    assert rows[0]["method"] == "EAP-IG-activations"
    assert rows[0]["status"] == "pass"
