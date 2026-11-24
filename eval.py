"""评估：划分训练/验证/测试集，用最佳验证模型报告测试困惑度。

- 名字去重后按 80%/10%/10% 划分训练/验证/测试集。
- 三个模型（前向 / 反向 / seq2seq）各自只用训练集训练，再在测试集上算：
  * loss：平均交叉熵
  * perplexity：exp(loss)，越小越好
- RNN 的指标是给定首字符后的预测；seq2seq 的指标是给定片段的占位符序列预测。
  两类任务的困惑度不应直接用于比较模型优劣。

运行：``python eval.py``（重新训练；checkpoint 和日志名称以 eval_ 开头）
"""

import math
import random

import torch

from names_rnn import EOS, HIDDEN_SIZE, RNN, build_vocab, load_names, train_rnn, rnn_loss, split_names
from seq2seq import SEP, SOS, make_pairs, train_seq2seq, seq2seq_loss

SEED = 0


def evaluate_rnn(model, sequences, char_to_idx):
    """返回 (loss, perplexity)。"""
    loss, _ = rnn_loss(model, sequences, char_to_idx)
    return loss, math.exp(loss)


def evaluate_seq2seq(encoder, decoder, pairs, char_to_idx):
    """返回 (loss, perplexity)。"""
    loss, _ = seq2seq_loss(encoder, decoder, pairs, char_to_idx)
    return loss, math.exp(loss)


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    names = load_names()
    train_names, val_names, test_names = split_names(names)
    print(f"训练 {len(train_names)} / 验证 {len(val_names)} / 测试 {len(test_names)}\n")
    print("注意：RNN 报告给定首字符后的困惑度；seq2seq 报告片段条件下占位符序列的困惑度，不能直接横向排名。\n")

    # 前向
    cti, _, vs = build_vocab(names, [EOS])
    forward = RNN(vs, HIDDEN_SIZE, vs)
    print("训练前向模型 ...")
    train_rnn(forward, train_names, cti, val_sequences=val_names, log_name="eval_forward")
    loss, ppl = evaluate_rnn(forward, test_names, cti)
    print(f"  前向   测试 loss={loss:.4f}  perplexity={ppl:.2f}\n")

    # 反向
    backward = RNN(vs, HIDDEN_SIZE, vs)
    print("训练反向模型 ...")
    train_rnn(backward, [n[::-1] for n in train_names], cti,
              val_sequences=[n[::-1] for n in val_names], log_name="eval_backward")
    loss, ppl = evaluate_rnn(backward, [n[::-1] for n in test_names], cti)
    print(f"  反向   测试 loss={loss:.4f}  perplexity={ppl:.2f}\n")

    # seq2seq
    cti_s, _, vs_s = build_vocab(names, [EOS, SOS, SEP])
    train_pairs = make_pairs(train_names)
    val_pairs = make_pairs(val_names)
    test_pairs = make_pairs(test_names)
    encoder = RNN(vs_s, HIDDEN_SIZE, vs_s)
    decoder = RNN(vs_s, HIDDEN_SIZE, vs_s)
    print("训练 seq2seq ...")
    train_seq2seq(encoder, decoder, train_pairs, cti_s,
                  val_pairs=val_pairs, log_name="eval_seq2seq")
    loss, ppl = evaluate_seq2seq(encoder, decoder, test_pairs, cti_s)
    print(f"  seq2seq 测试 loss={loss:.4f}  perplexity={ppl:.2f}")


if __name__ == "__main__":
    main()
