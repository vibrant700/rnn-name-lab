"""用字符级 RNN 生成英文名字（训练部分：前向 + 反向）。

- 前向 RNN：从左往右，给定开头补全后面。
- 反向 RNN：从右往左（训练在反转的名字上），给定结尾补全前面。
两个模型共用同一个 RNN 类，只是训练数据一个是 names、一个是 names[::-1]。
训练时保存验证损失最低的 best.pt，预测统一在 generate.py 里。

运行：``python names_rnn.py``
"""

import argparse
import json
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.nn as nn

EOS = "."            # 名字结束符（名字里不会出现这个字符）
HIDDEN_SIZE = 128    # 隐状态维度
EPOCHS = 50          # 训练轮数（想效果更好可调大，但更慢）
LEARNING_RATE = 0.005

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "names.txt")
CKPT_DIR = os.path.dirname(os.path.abspath(__file__))


# 数据
def load_names():
    """读取名字，统一转小写（减小词表），跳过空行。"""
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names


def build_vocab(names, special=()):
    """收集所有出现过的字符 + 特殊符，建立 字符<->下标 双向映射。"""
    chars = sorted(set("".join(names)))   # 排序保证每次结果一致
    chars.extend(special)                 # 追加特殊符（EOS、SOS 等）
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(chars)


def split_names(names, val_ratio=0.1, test_ratio=0.1, seed=0):
    """去重后按固定种子打乱，划分训练/验证/测试集，默认比例 80%/10%/10%。"""
    names = list(dict.fromkeys(names))
    random.Random(seed).shuffle(names)
    n_test = int(len(names) * test_ratio)
    n_val = int(len(names) * val_ratio)
    test = names[:n_test]              # 前 n_test 个
    val = names[n_test:n_test + n_val]  # 接下来 n_val 个
    train = names[n_test + n_val:]      # 剩下的
    return train, val, test


def training_paths(name, log_path=None, checkpoint_dir=None):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")  # UTC+8 时间戳
    log_path = Path(log_path or Path(CKPT_DIR) / "logs" / f"{name}_{stamp}.json")
    checkpoint_dir = Path(checkpoint_dir or log_path.with_suffix(""))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"训练日志：{log_path}", flush=True)
    print(f"最佳模型：{checkpoint_dir / 'best.pt'}", flush=True)
    return log_path, checkpoint_dir / "best.pt"


def save_training_log(path, name, config, history):
    data = {"model": name, "config": config, "history": history}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


# 模型
class RNN(nn.Module):
    """手写的字符级 RNN：输入一个字符，输出下一个字符的得分。

    每一步：hidden = tanh(W_ih·[x, hidden])，output = W_ho·[x, hidden]。
    前向、反向、以及 seq2seq 的编码器/解码器都复用这个类。
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # 输入+旧隐状态 -> 新隐状态
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # 输入+旧隐状态 -> 输出

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)   # 拼接输入和隐状态
        hidden = torch.tanh(self.i2h(combined))    # 更新隐状态
        output = self.i2o(combined)                # 输出得分（softmax 交给损失/采样）
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def char_to_tensor(c, char_to_idx):
    """把单个字符转成 one-hot 向量。"""
    t = torch.zeros(1, len(char_to_idx))
    t[0][char_to_idx[c]] = 1.0
    return t


def rnn_sequence_loss(model, seq, char_to_idx, criterion):
    hidden = model.init_hidden()
    loss = 0.0
    for c, target in zip(seq, seq[1:] + EOS):
        output, hidden = model(char_to_tensor(c, char_to_idx), hidden)
        loss = loss + criterion(output, torch.tensor([char_to_idx[target]]))
    return loss


def rnn_loss(model, sequences, char_to_idx):
    """固定参数，计算逐字符平均交叉熵。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for seq in sequences:
            total_loss += rnn_sequence_loss(model, seq, char_to_idx, criterion).item()
            total_tokens += len(seq)
    return total_loss / total_tokens, total_tokens


# 训练（通用：喂什么序列就学什么方向）
def train_rnn(model, sequences, char_to_idx, *, val_sequences, epochs=EPOCHS,
              learning_rate=LEARNING_RATE, log_name="rnn", log_path=None, seed=0,
              checkpoint_dir=None):
    """从头训练，逐轮记录训练/验证 loss，保存并载入最佳验证模型。"""
    sequences, val_sequences = list(sequences), list(val_sequences)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    config = dict(epochs=epochs, learning_rate=learning_rate, samples=len(sequences),
                  val_samples=len(val_sequences),
                  vocab_size=len(char_to_idx), hidden_size=model.hidden_size,
                  shuffle_seed=seed, torch_seed=torch.initial_seed(), gradient_clip=5.0)
    log_path, checkpoint_path = training_paths(log_name, log_path, checkpoint_dir)
    history = []
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        total_loss, total_tokens = 0.0, 0
        for index in rng.sample(range(len(sequences)), len(sequences)):
            seq = sequences[index]
            loss = rnn_sequence_loss(model, seq, char_to_idx, criterion)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            total_tokens += len(seq)

        train_loss = total_loss / total_tokens
        val_loss, val_tokens = rnn_loss(model, val_sequences, char_to_idx)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_tokens=total_tokens, val_tokens=val_tokens,
                            seconds=time.perf_counter() - started))
        data = save_training_log(log_path, log_name, config, history)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "models": {"rnn": model.state_dict()},
                        "best_val_loss": best_val_loss, "config": config,
                        "char_to_idx": char_to_idx}, checkpoint_path)
        print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["models"]["rnn"])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--direction", choices=["both", "forward", "backward"], default="both")
    args = parser.parse_args()
    torch.manual_seed(0)

    # 1) 读数据、建词表
    names = load_names()
    train_names, val_names, test_names = split_names(names)
    char_to_idx, _, vocab_size = build_vocab(names, [EOS])
    print(f"共 {len(names)} 个名字，词表大小 {vocab_size}")
    print(f"去重划分：训练 {len(train_names)} / 验证 {len(val_names)} / 测试 {len(test_names)}")

    # 2) 训练前向、反向两个模型
    if args.direction in ("both", "forward"):
        forward = RNN(vocab_size, HIDDEN_SIZE, vocab_size)
        print("训练前向模型 ...")
        train_rnn(forward, train_names, char_to_idx, val_sequences=val_names,
                  epochs=args.epochs, log_name="forward")
    if args.direction in ("both", "backward"):
        backward = RNN(vocab_size, HIDDEN_SIZE, vocab_size)
        print("训练反向模型 ...")
        train_rnn(backward, [n[::-1] for n in train_names], char_to_idx,
                  val_sequences=[n[::-1] for n in val_names], epochs=args.epochs,
                  log_name="backward")
