"""将 best.pt 中的实际 RNN 权重导出为 GitHub Pages 可用的静态资源。"""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "assets"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    binary = bytearray()
    manifest = {"version": 1, "eos": ".", "sos": "^", "sep": "#", "models": {}}
    training = {"models": {}, "data": {}}
    for task in ("forward", "backward", "seq2seq"):
        path = max((ROOT / "logs").glob(f"{task}_*/best.pt"), key=lambda p: p.stat().st_mtime)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        chars = [c for c, _ in sorted(checkpoint["char_to_idx"].items(), key=lambda pair: pair[1])]
        for key, state in checkpoint["models"].items():
            name = task if key == "rnn" else key
            entry = {"chars": chars, "hiddenSize": checkpoint["config"]["hidden_size"], "tensors": {}}
            for parameter in ("i2h.weight", "i2h.bias", "i2o.weight", "i2o.bias"):
                tensor = state[parameter].detach().cpu().contiguous().numpy().astype("<f4")
                entry["tensors"][parameter] = {"offset": len(binary) // 4, "shape": list(tensor.shape)}
                binary.extend(tensor.tobytes())
            manifest["models"][name] = entry
        log = json.loads(path.parent.with_suffix(".json").read_text(encoding="utf-8"))
        if task == "forward":
            training["data"].update(train=checkpoint["config"]["samples"], validation=checkpoint["config"]["val_samples"])
        training["models"][task] = {
            "bestEpoch": checkpoint["epoch"],
            "bestValLoss": checkpoint["best_val_loss"],
            "history": [{k: row[k] for k in ("epoch", "train_loss", "val_loss")} for row in log["history"]],
        }
    training["test"] = json.loads((ROOT / "logs" / "test_metrics.json").read_text(encoding="utf-8"))
    training["data"]["test"] = training["test"]["test_names"]
    names = [line.strip().lower() for line in (ROOT / "names.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    training["data"].update(total=len(names), unique=len(set(names)))
    (OUTPUT / "weights.bin").write_bytes(binary)
    (OUTPUT / "models.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUTPUT / "training.json").write_text(json.dumps(training, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(manifest['models'])} RNNs, {len(binary):,} bytes → {OUTPUT}")


if __name__ == "__main__":
    main()
