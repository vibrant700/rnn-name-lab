"""预测（生成）：加载训练好的模型，按三种模式生成名字。

模式：
  1 = 给开头补后面（前向 RNN）
  2 = 给结尾补前面（反向 RNN）
  3 = 给中间补两边（seq2seq）

运行：``python generate.py``（先跑 names_rnn.py 和 seq2seq.py 生成 checkpoint）
"""

from pathlib import Path

import torch

from names_rnn import EOS, HIDDEN_SIZE, RNN, build_vocab, char_to_tensor, load_names
from seq2seq import SOS, SEP

CKPT_DIR = Path(__file__).resolve().parent / "logs"


def load_best_checkpoint(name):
    """加载该模型目录下最新一次训练保存的 best.pt。"""
    path = max(CKPT_DIR.glob(f"{name}_*/best.pt"), key=lambda p: p.stat().st_mtime)
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_fragment(value, char_to_idx):
    """输入必须非空，只允许数据字符，不允许模型控制符。"""
    if not value:
        raise ValueError("请输入至少一个字符")
    if any(c not in char_to_idx or c in (EOS, SOS, SEP) for c in value):
        raise ValueError("含不支持的字符；请使用数据集中的字母、空格、连字符或撇号")


# 生成（前向/反向通用）：从 prefix 往后生成
def generate_rnn(model, prefix, char_to_idx, idx_to_char, max_len=20, decoding="sample"):
    """给定前缀，随机采样或贪心选择后续字符，直到结束符。"""
    validate_fragment(prefix, char_to_idx)
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden()
        output = None
        for c in prefix:   # 喂入前缀字符
            output, hidden = model(char_to_tensor(c, char_to_idx), hidden)

        result = prefix
        for _ in range(max_len):
            if decoding == "greedy":
                idx = output.argmax(dim=1).item()
            else:
                prob = torch.softmax(output, dim=1)
                idx = torch.multinomial(prob, 1).item()
            c = idx_to_char[idx]
            if c == EOS:
                break
            result += c
            output, hidden = model(char_to_tensor(c, char_to_idx), hidden)
    return result


# 生成（seq2seq）：给中间片段，生成完整名字
def generate_middle(encoder, decoder, fragment, char_to_idx, idx_to_char, max_len=20, decoding="sample"):
    """约束解码：一个 SEP，左右各至少一个数据字符；max_len 为完整名字上限。"""
    validate_fragment(fragment, char_to_idx)
    if max_len < len(fragment) + 2:
        raise ValueError(f"中间片段过长：完整名字上限为 {max_len}，需为两侧各留一个字符")
    # SEP 替换成 fragment 后，完整名字仍不超过 max_len。
    template_budget = max_len - len(fragment) + 1
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # 编码
        enc_hidden = encoder.init_hidden()
        for c in fragment:
            _, enc_hidden = encoder(char_to_tensor(c, char_to_idx), enc_hidden)

        # 解码
        dec_hidden = enc_hidden
        prev = char_to_tensor(SOS, char_to_idx)
        result = ""
        sep_position = None
        for step in range(template_budget):
            output, dec_hidden = decoder(prev, dec_hidden)
            logits = output.clone()
            logits[0, char_to_idx[SOS]] = -torch.inf
            if sep_position is None:
                logits[0, char_to_idx[EOS]] = -torch.inf
                if not result:
                    logits[0, char_to_idx[SEP]] = -torch.inf
                # 最迟在倒数第二步放 SEP，最后一步留给右侧字符。
                if step == template_budget - 2:
                    logits.fill_(-torch.inf)
                    logits[0, char_to_idx[SEP]] = 0.0
            else:
                logits[0, char_to_idx[SEP]] = -torch.inf
                if len(result) == sep_position + 1:
                    logits[0, char_to_idx[EOS]] = -torch.inf
            if decoding == "greedy":
                idx = logits.argmax(dim=1).item()
            else:
                prob = torch.softmax(logits, dim=1)
                idx = torch.multinomial(prob, 1).item()
            c = idx_to_char[idx]
            if c == EOS:
                break
            if c == SEP:
                sep_position = len(result)
            result += c
            prev = char_to_tensor(c, char_to_idx)
    return result.replace(SEP, fragment)   # 占位符换回输入的片段


def main():
    names = load_names()
    # 前向/反向使用数据字符 + EOS；seq2seq 额外使用 SOS 和 SEP。
    cti, itc, vs = build_vocab(names, [EOS])
    cti_s, itc_s, vs_s = build_vocab(names, [EOS, SOS, SEP])

    # 加载四个模型
    forward = RNN(vs, HIDDEN_SIZE, vs)
    backward = RNN(vs, HIDDEN_SIZE, vs)
    encoder = RNN(vs_s, HIDDEN_SIZE, vs_s)
    decoder = RNN(vs_s, HIDDEN_SIZE, vs_s)
    forward.load_state_dict(load_best_checkpoint("forward")["models"]["rnn"])
    backward.load_state_dict(load_best_checkpoint("backward")["models"]["rnn"])
    seq_checkpoint = load_best_checkpoint("seq2seq")
    encoder.load_state_dict(seq_checkpoint["models"]["encoder"])
    decoder.load_state_dict(seq_checkpoint["models"]["decoder"])

    # 交互式生成
    decoding = "greedy" if input("生成方式：1=随机采样  2=贪心解码（默认1）：").strip() == "2" else "sample"
    count = 1 if decoding == "greedy" else 5
    print("模式：1=给开头  2=给结尾  3=给中间  回车=退出")
    while True:
        mode = input("模式：").strip()
        if mode == "1":
            prefix = input("开头：").strip().lower()
            try:
                validate_fragment(prefix, cti)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            for _ in range(count):
                print("  " + generate_rnn(forward, prefix, cti, itc, decoding=decoding))
        elif mode == "2":
            suffix = input("结尾：").strip().lower()
            try:
                validate_fragment(suffix, cti)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            for _ in range(count):
                rev = generate_rnn(backward, suffix[::-1], cti, itc, decoding=decoding)
                print("  " + rev[::-1])
        elif mode == "3":
            fragment = input("中间：").strip().lower()
            try:
                validate_fragment(fragment, cti_s)
                if len(fragment) > 18:
                    raise ValueError("中间片段最多 18 个字符，需为两侧各留一个字符")
            except ValueError as exc:
                print(f"  {exc}")
                continue
            for _ in range(count):
                print("  " + generate_middle(encoder, decoder, fragment, cti_s, itc_s, decoding=decoding))
        else:
            break


if __name__ == "__main__":
    main()
