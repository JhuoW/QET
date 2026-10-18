"""Download the ToolHop data at its pinned dataset commit into data/toolhop/ (`make fetch`; D-035) and check its hash:

    python scripts/fetch_toolhop.py

The dataset (bytedance-research/ToolHop) is Apache-2.0 and not gated. Its tools are code that the evaluation runs in
a sandbox (src/qet/envs/toolhop.py); this script only downloads and checks files.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = ["data/ToolHop.json", "LICENSE", "README.md", "code/evaluation_open.py"]


def main() -> None:
    from huggingface_hub import hf_hub_download

    from qet.envs.toolhop import DATA, DATA_SHA256, DATASET_COMMIT

    out = REPO / "data" / "toolhop"
    for f in FILES:
        hf_hub_download("bytedance-research/ToolHop", f, repo_type="dataset", revision=DATASET_COMMIT, local_dir=out)
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise SystemExit(f"{DATA} has sha256 {digest}, not the pinned {DATA_SHA256}")
    print(f"{DATA.relative_to(REPO)} at {DATASET_COMMIT[:7]}: sha256 {digest} (pinned)")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()
