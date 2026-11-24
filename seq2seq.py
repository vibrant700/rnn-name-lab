"""给中间字母补全（训练部分）：seq2seq 编码器-解码器。

- 训练数据：对每个名字随机取一段中间连续字母，配成 (片段, 完整名字)。
- 编码器读片段得到上下文，解码器从上下文生成完整名字（片段落在中间）。
- 训练时保存验证损失最低的 best.pt，预测统一在 generate.py 里。

运行：``python seq2seq.py``
"""

import argparse
import random
import time

import torch
import torch.nn as nn

from names_rnn import (EOS, HIDDEN_SIZE, RNN, build_vocab, char_to_tensor, load_names,
                       split_names, training_paths, save_training_log)

SOS = "^"            # 起始符（解码器的开始信号，名字里不会出现）
SEP = "#"            # 片段占位符（生成时替换回输入的片段）
EPOCHS = 50
LEARNING_RATE = 0.005


def make_pairs(names, per_name=1):
    """对每个名字随机取一段中间连续字母，构成 (片段, 目标) 训练样本。

    目标 = 名字里把片段换成 SEP 占位符，例如 "alice" 的片段 "lic" -> "a#e"。
    """
    pairs = []
    for name in names:
        n = len(name)
        if n < 3:          # 太短没法取中间
            continue
        for _ in range(per_name):
            start = random.randint(1, n - 2)         # 片段不碰第一个字符
            end = random.randint(start + 1, n - 1)   # 片段不碰最后一个字符
            pairs.append((name[start:end], name[:start] + SEP + name[end:]))
    return pairs


def seq2seq_sequence_loss(encoder, decoder, fragment, target, char_to_idx, criterion):
    hidden = encoder.init_hidden()
    for c in fragment:
        _, hidden = encoder(char_to_tensor(c, char_to_idx), hidden)
    prev = char_to_tensor(SOS, char_to_idx)
    loss = 0.0
    for c in target + EOS:
        output, hidden = decoder(prev, hidden)
        loss = loss + criterion(output, torch.tensor([char_to_idx[c]]))
        prev = char_to_tensor(c, char_to_idx)
    return loss


def seq2seq_loss(encoder, decoder, pairs, char_to_idx):
    """teacher forcing 下的逐字符平均交叉熵。"""
    encoder.eval()
    decoder.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for fragment, target in pairs:
            total_loss += seq2seq_sequence_loss(
                encoder, decoder, fragment, target, char_to_idx, criterion
            ).item()
            total_tokens += len(target) + 1
    return total_loss / total_tokens, total_tokens


# 训练
def train_seq2seq(encoder, decoder, pairs, char_to_idx, *, val_pairs, epochs=EPOCHS,
                  learning_rate=LEARNING_RATE, log_name="seq2seq", log_path=None, seed=0,
                  checkpoint_dir=None):
    """从头训练，逐轮记录训练/验证 loss，保存并载入最佳验证模型。"""
    pairs, val_pairs = list(pairs), list(val_pairs)
    criterion = nn.CrossEntropyLoss()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    rng = random.Random(seed)
    config = dict(epochs=epochs, learning_rate=learning_rate, samples=len(pairs),
                  val_samples=len(val_pairs),
                  vocab_size=len(char_to_idx), hidden_size=encoder.hidden_size,
                  shuffle_seed=seed, torch_seed=torch.initial_seed(), gradient_clip=5.0)
    log_path, checkpoint_path = training_paths(log_name, log_path, checkpoint_dir)
    history = []
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        started = time.perf_counter()
        total_loss, total_tokens = 0.0, 0
        for index in rng.sample(range(len(pairs)), len(pairs)):
            fragment, target = pairs[index]
            loss = seq2seq_sequence_loss(encoder, decoder, fragment, target, char_to_idx, criterion)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            total_tokens += len(target) + 1

        train_loss = total_loss / total_tokens
        val_loss, val_tokens = seq2seq_loss(encoder, decoder, val_pairs, char_to_idx)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_tokens=total_tokens, val_tokens=val_tokens,
                            seconds=time.perf_counter() - started))
        data = save_training_log(log_path, log_name, config, history)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch,
                        "models": {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                        "best_val_loss": best_val_loss, "config": config,
                        "char_to_idx": char_to_idx}, checkpoint_path)
        print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(best["models"]["encoder"])
    decoder.load_state_dict(best["models"]["decoder"])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    # 1) 读数据、建词表（比前向/反向多 SOS 和 SEP 两个特殊符）
    names = load_names()
    train_names, val_names, test_names = split_names(names)
    char_to_idx, _, vocab_size = build_vocab(names, [EOS, SOS, SEP])
    print(f"共 {len(names)} 个名字，词表大小 {vocab_size}")
    print(f"去重划分：训练 {len(train_names)} / 验证 {len(val_names)} / 测试 {len(test_names)}")

    # 2) 造数据、训练编码器-解码器
    pairs = make_pairs(train_names)
    val_pairs = make_pairs(val_names)
    encoder = RNN(vocab_size, HIDDEN_SIZE, vocab_size)
    decoder = RNN(vocab_size, HIDDEN_SIZE, vocab_size)
    print(f"训练样本 {len(pairs)} 个，训练 seq2seq ...")
    train_seq2seq(encoder, decoder, pairs, char_to_idx, val_pairs=val_pairs,
                  epochs=args.epochs)
