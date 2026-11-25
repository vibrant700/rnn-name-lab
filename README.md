# Name Lab · 字母实验室

给几个字母，让字符级 RNN 把它们写成名字。

**[在线体验](https://vibrant700.github.io/rnn-name-lab/)** · [代码仓库](https://github.com/vibrant700/rnn-name-lab)

支持开头、结尾和中间片段补全。先生成六个候选名字，由用户点选喜欢的一个，再在动态框中逐字展示它的生成记录，同步更新全部 30/32 个字符的原始预测概率，并突出 Top-5 和实际选中字符。支持暂停、单步、调整速度和重播，还能下载过程图、查看训练与验证 loss 曲线。页面打开时不自动生成或展示过程。

候选名字由浏览器模型实际生成，每一步都保留完整概率；用户选择后，播放对应名字的真实生成记录，暂停和单步不会改变已采样的名字。

网页提供两种生成方式：**随机采样**按概率选择字符，保留随机性滑块并生成六个候选；**每步选最大概率**使用贪心解码，停用随机性设置，相同输入只展示一个确定结果。中间补全在满足片段约束的允许字符中选最大概率，可能不等于原始 Top-5 的第一名，逐步说明会明确标注。

## 在 GitHub Pages 上运行真实模型

网站是无第三方依赖的 HTML/CSS/JavaScript。训练好的四个网络以 Float32 静态权重发布，约 400 KB；浏览器中的 Web Worker 直接执行原来的线性层和 tanh，生成过程无需后端、API 密钥或登录。

前向和反向模型分别负责从左向右、从右向左补全；中间补全使用编码器/解码器生成包含一个占位符的序列，再替换回输入片段，并约束两侧各有字符。

当前模型的每一步严格对应 Python 实现：

```text
combined = [one_hot(character), old_hidden]
new_hidden = tanh(i2h(combined))
logits = i2o(combined)
```

输出层使用的是旧隐状态。网页推理没有改成另一种 RNN。Top-5 展示温度调整和补全约束之前的原始模型概率，实际采样结果可能不在 Top-5 内。达到长度上限时会标记截断，不伪装成模型预测了结束符。

## 本地预览

```bash
python -m http.server 8000 --directory docs
```

打开 `http://localhost:8000`。仓库已包含浏览器运行所需权重和曲线数据，不需要先安装 PyTorch。

## 重新训练与导出

```bash
python -m pip install -r requirements.txt
python names_rnn.py --epochs 50
python seq2seq.py --epochs 50
python export_web.py
python -m http.server 8000 --directory docs
```

也可以在三个终端中分别运行前向、反向和 seq2seq：

```bash
python names_rnn.py --direction forward --epochs 50
python names_rnn.py --direction backward --epochs 50
python seq2seq.py --epochs 50
```

训练使用 CPU，隐藏状态维度为 128。前向和反向模型的学习率为 0.005，Seq2seq 为 0.001。每轮保存训练/验证 loss，只保留验证损失最低的 `best.pt`，不提供断点续训。原始训练目录 `logs/` 不纳入版本管理；网页公开资源只含必要模型权重、训练曲线和评估数据，不包含本地机器路径、优化器状态或样本副本。

`export_web.py` 读取各模型最新运行的 `best.pt` 和对应 JSON 日志。若重新计算测试结果，可运行 `python eval.py`；该脚本会重新训练并输出测试指标。仓库当前发布的测试指标保留在 `docs/assets/training.json`。

## 数据与效果

数据来自 [CMU 英文名字语料](http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/)。本地文件有 7,944 条记录，转小写去重后有 7,575 个名字；固定划分为 6,061 个训练名字、757 个验证名字和 757 个测试名字。

| 模型 | 最佳轮次 | 验证 loss | 测试 loss |
| --- | ---: | ---: | ---: |
| 前向 RNN | 17 | 2.2476 | 2.2703 |
| 反向 RNN | 41 | 2.4225 | 2.4104 |
| Seq2seq | 9 | 2.0497 | 2.0539 |

Seq2seq 预测的是给定片段条件下的占位符序列，与另外两个任务的 loss 不直接比较。将学习率从 0.005 降到 0.001 后，完整重训 50 轮，最佳验证模型出现在第 9 轮；后期验证损失回升，故保留第 9 轮权重。在本次相同测试片段上，旧模型 loss 为 2.3107，新模型为 2.0539，降低约 11.1%。生成的名字是随机组合，不保证是真实名字或符合所有语言习惯。

## 验证与发布

```bash
npm run check
python tests/verify_inference.py
```

数值测试使用相同导出权重，对照 224 个 PyTorch/JavaScript 时刻的隐状态和 logits，再验证 360 次随机采样的片段保留、长度、EOS 标记和随机种子复现；另有 120 个贪心用例验证允许字符中的最大概率选择，以及对随机种子和温度的不变性。

GitHub Pages 设置：**Settings → Pages → Deploy from a branch → main / docs**。所有资源使用相对路径，支持项目站点的 `/rnn-name-lab/` 路径。

## 文件

- `names_rnn.py`、`seq2seq.py`：训练模型、记录训练/验证损失、保存最佳模型。
- `generate.py`：Python 交互式生成。
- `eval.py`：训练、验证选择和测试评估。
- `export_web.py`：导出浏览器静态资源。
- `docs/`：GitHub Pages 网站、推理代码与发布权重。
- `tests/`：Python 和 JavaScript 推理一致性验证。
