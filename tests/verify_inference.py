"""用网页导出的同一组权重，对照 PyTorch 和 JavaScript 的逐步输出。"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from names_rnn import RNN, char_to_tensor


torch.set_num_threads(1)
manifest = json.loads((ROOT / "docs/assets/models.json").read_text(encoding="utf-8"))
weights = torch.frombuffer(bytearray((ROOT / "docs/assets/weights.bin").read_bytes()), dtype=torch.float32)
fixtures = []
for name, metadata in manifest["models"].items():
    size = len(metadata["chars"])
    model = RNN(size, metadata["hiddenSize"], size)
    state = {}
    for key, tensor in metadata["tensors"].items():
        count = 1
        for dimension in tensor["shape"]:
            count *= dimension
        state[key] = weights[tensor["offset"]:tensor["offset"] + count].reshape(tensor["shape"])
    model.load_state_dict(state)
    mapping = {c: i for i, c in enumerate(metadata["chars"])}
    for sequence in ("alice", "robert", "o'neil", "mary-jane", "anna maria", "abcdefghijklmnopqrst"):
        hidden = model.init_hidden()
        steps = []
        with torch.no_grad():
            for c in sequence:
                output, hidden = model(char_to_tensor(c, mapping), hidden)
                steps.append({"char": c, "logits": output[0].tolist(), "hidden": hidden[0].tolist()})
        fixtures.append({"model": name, "steps": steps})
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "fixtures.json"
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    subprocess.run(["node", str(ROOT / "tests/verify-inference.mjs"), str(path)], cwd=ROOT, check=True)
