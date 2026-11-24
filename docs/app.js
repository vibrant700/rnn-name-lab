const $ = id => document.getElementById(id);
const modes = {
  prefix: { label: "名字的开头", hint: "输入一到几个英文字母，补全后面的可能。", examples: ["al", "mar", "el"], direction: "→", description: "给定开头，前向 RNN 从左向右续写。" },
  suffix: { label: "名字的结尾", hint: "留一个你喜欢的结尾，向前寻找一个名字。", examples: ["na", "son", "ia"], direction: "←", description: "给定结尾，反向 RNN 从右向左补全。" },
  middle: { label: "名字中间的片段", hint: "保留这几个字母，让模型探索它的左右两侧。", examples: ["ann", "li", "ari"], direction: "↔", description: "编码器读取片段，解码器生成两侧，再将片段嵌入名字。" },
};
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let mode = "prefix";
let ready = false;
let loading = true;
let busy = false;
let results = [];
let selected = 0;
let stepIndex = 0;
let batch = 0;
let playTimer;
let streaming = false;
let playing = false;
let requestId = 0;
let training;

const percent = value => `${(value * 100).toFixed(1)}%`;
const letterLabel = char => ({ ".": "EOS", "^": "开始", "#": "片段", " ": "空格" })[char] || char;
const titleCase = text => text.charAt(0).toUpperCase() + text.slice(1);
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function highlightName(result) {
  const container = document.createDocumentFragment();
  const name = titleCase(result.name);
  const start = result.mode === "prefix" ? 0 : result.mode === "suffix" ? name.length - result.fragment.length : result.steps.find(step => step.char === "#")?.templateBefore.length;
  if (start === undefined) { container.append(document.createTextNode(name)); return container; }
  container.append(document.createTextNode(name.slice(0, start)));
  container.append(node("mark", "fragment-mark", name.slice(start, start + result.fragment.length)));
  container.append(document.createTextNode(name.slice(start + result.fragment.length)));
  return container;
}

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function setBusy(value) {
  busy = value;
  $("generate-button").disabled = value;
  $("generate-label").textContent = value ? "字母正在相遇…" : "生成并逐字演示";
  document.querySelectorAll("[data-mode]").forEach(button => button.disabled = value);
  $("name-grid").setAttribute("aria-busy", String(value));
}

function updateMode(next, generate = true) {
  stopPlayback();
  mode = next;
  const config = modes[mode];
  document.querySelectorAll("[data-mode]").forEach(button => {
    const active = button.dataset.mode === mode;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  $("input-panel").setAttribute("aria-labelledby", `mode-${mode}`);
  $("fragment-label").textContent = config.label;
  $("fragment-hint").textContent = config.hint;
  $("input-direction").textContent = config.direction;
  $("fragment").value = config.examples[0];
  $("fragment").maxLength = mode === "middle" ? 18 : 19;
  $("examples").replaceChildren(...config.examples.map(example => {
    const button = node("button", "", example);
    button.type = "button";
    button.addEventListener("click", () => { $("fragment").value = example; requestGeneration(true); });
    return button;
  }));
  if (generate && ready) requestGeneration(true);
}

function requestGeneration(reveal = false) {
  if (busy) return;
  if (!ready) {
    if (!loading) { loading = true; $("generate-button").disabled = true; worker.postMessage({ type: "load" }); status("正在重新加载模型…"); }
    return;
  }
  const fragment = $("fragment").value.trim().toLowerCase();
  if (!fragment || !/^[a-z '\-]+$/.test(fragment)) {
    status("请先输入英文名字片段，支持字母、空格、连字符和撇号。", true);
    $("fragment").focus();
    return;
  }
  $("fragment").value = fragment;
  stopPlayback();
  streaming = false;
  setBusy(true);
  if (reveal) $("process").scrollIntoView({ behavior: "smooth", block: "start" });
  status("正在逐字生成，第一个名字完成后会提供更多候选。");
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  worker.postMessage({ type: "generate", id: ++requestId, seed, options: { mode, fragment, temperature: Number($("temperature").value), maxLength: 20 } });
}

worker.onmessage = ({ data }) => {
  if (data.id !== undefined && data.id !== requestId) return;
  if (data.type === "ready") {
    ready = true;
    loading = false;
    requestGeneration();
  } else if (data.type === "started") {
    streaming = true;
    setBusy(false);
    results = [{ ...data.options, name: data.options.mode === "middle" ? "" : data.options.fragment,
      steps: [], terminated: false, truncated: false, pending: true }];
    selected = 0;
    chooseResult(0, false);
    playing = true;
    updatePlaybackButton();
    $("name-grid").replaceChildren(node("div", "empty-card streaming-placeholder", "正在写下第一个名字…"));
    $("favorite-name").textContent = titleCase(results[0].name) || "·";
    $("generation-phase").textContent = "模型正在预测第一笔";
    $("probability-bars").replaceChildren();
    $("all-probabilities").replaceChildren();
  } else if (data.type === "step") {
    const result = results[0];
    result.steps.push(data.step);
    result.name = data.step.after;
    result.terminated = data.step.char === ".";
    renderTimeline();
    renderStep(result.steps.length - 1);
    if (playing) scheduleNext();
  } else if (data.type === "results") {
    streaming = false;
    stopPlayback();
    setBusy(false);
    results = data.results;
    batch++;
    $("batch-label").textContent = `BATCH ${String(batch).padStart(3, "0")}`;
    $("generation-timing").textContent = `${results[0].steps.length} 步预测 · 逐字演示完成`;
    status("生成完成。可重播、单步查看，或选择另一个候选名字。");
    selected = 0;
    renderCards();
    chooseResult(selected, false);
    renderStep(results[selected].steps.length - 1);
  } else if (data.type === "error") {
    streaming = false;
    stopPlayback();
    setBusy(false);
    loading = false;
    status(data.message, true);
    if (!ready) $("generate-label").textContent = "重新加载模型";
  }
};
worker.onerror = () => {
  streaming = false;
  stopPlayback();
  setBusy(false);
  loading = false;
  status("模型暂时无法加载，请刷新页面重试。", true);
};

function renderCards() {
  $("name-grid").replaceChildren(...results.map((result, index) => {
    const card = node("button", "name-card");
    card.type = "button";
    card.style.animationDelay = `${index * 35}ms`;
    card.setAttribute("aria-pressed", String(index === selected));
    card.setAttribute("aria-label", `查看名字 ${result.name} 的生成过程`);
    const top = node("span", "card-top");
    top.append(node("span", "", `0${index + 1}`), node("span", "card-star", index === selected ? "✦" : "✧"));
    const name = node("span", "card-name");
    name.append(highlightName(result));
    card.append(top, name, node("span", "card-caption", index === selected ? "这个，我喜欢" : "看看它的故事 ↗"));
    card.addEventListener("click", () => { selected = index; renderCards(); chooseResult(index); $("process").scrollIntoView({ behavior: "smooth", block: "start" }); });
    return card;
  }));
}

function chooseResult(index, autoplay = true) {
  stopPlayback();
  selected = index;
  stepIndex = 0;
  const result = results[index];
  $("process-placeholder").hidden = true;
  $("process-content").hidden = false;
  $("save-image").disabled = Boolean(result.pending);
  $("copy-name").disabled = Boolean(result.pending);
  $("restart-process").disabled = Boolean(result.pending);
  $("story-description").textContent = `${modes[result.mode].description} 输入片段「${result.fragment}」。`;
  $("end-status").textContent = result.pending ? "◌ 模型按步预测，概率随每个字符同步更新" : result.terminated ? "◉ 模型预测了结束符 EOS，名字生成完成" : "◌ 已到长度上限，模型尚未预测结束符";
  $("end-status").classList.toggle("truncated", !result.terminated);
  $("prediction-direction").textContent = result.mode === "suffix" ? "向左补一笔" : "下一笔";
  renderTimeline();
  renderStep(0);
  if (autoplay && result.steps.length) startPlayback();
}

function renderTimeline() {
  $("timeline").replaceChildren(...results[selected].steps.map((step, index) => {
    const button = node("button", `timeline-step${".^# ".includes(step.char) ? " special" : ""}`);
    button.type = "button";
    button.append(node("span", "", letterLabel(step.char)), node("small", "", String(index + 1).padStart(2, "0")));
    button.dataset.char = step.char;
    button.setAttribute("aria-label", `第 ${index + 1} 步：${letterLabel(step.char)}`);
    button.addEventListener("click", () => { stopPlayback(); renderStep(index); });
    return button;
  }));
}

function renderGrowingName(result, step) {
  const before = step.before;
  const after = step.after;
  const insertion = result.mode === "suffix" ? 0 : before.length;
  const added = after.length - before.length;
  const givenStart = result.mode === "prefix" ? 0 : result.mode === "suffix" ? after.length - result.fragment.length :
    step.char === "#" ? step.templateBefore.length : step.templateBefore.indexOf("#");
  $("favorite-name").replaceChildren(...[...titleCase(after)].map((char, index) => {
    const given = givenStart >= 0 && index >= givenStart && index < givenStart + result.fragment.length;
    return node("span", `live-letter${given ? " given" : ""}${added > 0 && index >= insertion && index < insertion + added ? " just-generated" : ""}`, char);
  }));
  if (step.char !== ".") $("favorite-name").append(node("span", "typing-cursor"));
  $("favorite-name").dataset.current = after;
  $("generation-phase").textContent = step.char === "." ? "预测到 EOS · 名字完成" :
    streaming ? (playing ? `正在生成 · 第 ${step.index} 笔` : `已暂停 · 第 ${step.index} 笔`) :
    playing ? `逐字回放 · 第 ${step.index} 笔` : `第 ${step.index} 步的生成结果`;
  $("generation-transition").textContent = `${step.before || "开始"} ${result.mode === "suffix" ? "←" : "→"} ${step.after || "开始"}${step.char === "." ? "  [EOS]" : ""}`;
  const complete = !result.pending && stepIndex === result.steps.length - 1;
  $("end-status").textContent = step.char === "." ? "◉ 模型预测了结束符 EOS，名字生成完成" :
    complete && result.truncated ? "◌ 已到长度上限，模型尚未预测结束符" : "◌ 每一笔对应当前的完整概率分布";
  $("end-status").classList.toggle("truncated", complete && result.truncated);
}

function renderDistribution(step) {
  const ranked = new Map(step.top5.map((candidate, index) => [candidate.char, index + 1]));
  const ordered = [...step.distribution].sort((a, b) => {
    const order = char => /^[a-z]$/.test(char) ? char.charCodeAt(0) : 1000 + step.distribution.findIndex(candidate => candidate.char === char);
    return order(a.char) - order(b.char);
  });
  $("all-probabilities").replaceChildren(...ordered.map(candidate => {
    const rank = ranked.get(candidate.char);
    const cell = node("div", `probability-cell${rank ? " top-five" : ""}${candidate.char === step.char ? " selected" : ""}`);
    cell.setAttribute("role", "listitem");
    cell.dataset.char = candidate.char;
    cell.dataset.probability = String(candidate.probability);
    cell.setAttribute("aria-label", `${letterLabel(candidate.char)}：${percent(candidate.probability)}${rank ? `，第 ${rank} 名` : ""}${candidate.char === step.char ? "，本步选中" : ""}`);
    cell.title = `${letterLabel(candidate.char)}：${(candidate.probability * 100).toPrecision(6)}%`;
    cell.style.setProperty("--probability", `${candidate.probability * 100}%`);
    cell.append(node("span", "cell-char", letterLabel(candidate.char)),
      node("span", "cell-probability", candidate.probability > 0 && candidate.probability < .0001 ? "<0.01%" : `${(candidate.probability * 100).toFixed(2)}%`));
    if (rank) cell.append(node("span", "cell-rank", String(rank)));
    return cell;
  }));
}

function renderStep(index) {
  stepIndex = index;
  const result = results[selected];
  const step = result.steps[index];
  if (!step) return;
  renderGrowingName(result, step);
  renderDistribution(step);
  $("step-before").textContent = step.before || "开始解码";
  $("chosen-letter").textContent = letterLabel(step.char);
  $("chosen-letter").style.fontSize = ".^# ".includes(step.char) ? "27px" : "42px";
  $("chosen-probability").textContent = `原始概率 ${percent(step.probability)}`;
  $("step-counter").textContent = `${String(index + 1).padStart(2, "0")} / ${String(result.steps.length).padStart(2, "0")}`;
  const max = Math.min(1, Math.ceil(step.top5[0].probability * 10) / 10);
  $("chart-max").textContent = `${Math.round(max * 100)}%`;
  $("probability-bars").replaceChildren(...step.top5.map(candidate => {
    const row = node("div", `probability-row${candidate.char === step.char ? " chosen" : ""}`);
    row.setAttribute("role", "listitem");
    const label = node("span", `probability-letter${".^# ".includes(candidate.char) ? " special" : ""}`, letterLabel(candidate.char));
    const track = node("div", "probability-track");
    const fill = node("div", "probability-fill");
    fill.style.width = `${candidate.probability / max * 100}%`;
    track.append(fill);
    row.append(label, track, node("span", "probability-value", percent(candidate.probability)));
    return row;
  }));
  $("step-note").textContent = step.char === "#" ? `这一笔嵌入输入片段「${result.fragment}」，原样保留。` : step.char === "." ? "这一笔是结束符，不会成为名字的一部分。" : !step.top5.some(candidate => candidate.char === step.char) ? `随机采样选中了「${step.char}」，它不在当前 Top-5 中。` : `这一步选中「${step.char}」，采样概率 ${percent(step.samplingProbability)}。`;
  [...$("timeline").children].forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
  if (playing) {
    const item = $("timeline").children[index];
    if (item) $("timeline").scrollLeft = Math.max(0, item.offsetLeft - $("timeline").offsetLeft - $("timeline").clientWidth / 2);
  }
}

function stopPlayback() {
  clearInterval(playTimer);
  clearTimeout(playTimer);
  playTimer = undefined;
  playing = false;
  updatePlaybackButton();
}

function updatePlaybackButton() {
  $("play-process").setAttribute("aria-pressed", String(playing));
  $("play-process").textContent = playing ? "Ⅱ 暂停" : streaming ? "▶ 继续生成" : "▶ 播放过程";
}

function scheduleNext(delay = Number($("playback-speed").value)) {
  clearTimeout(playTimer);
  playTimer = setTimeout(() => worker.postMessage({ type: "next", id: requestId }), delay);
}

function startPlayback() {
  playing = true;
  updatePlaybackButton();
  if (streaming) {
    if (results[selected].steps.length) renderStep(results[selected].steps.length - 1);
    scheduleNext();
    return;
  }
  if (stepIndex >= results[selected].steps.length - 1) renderStep(0);
  playTimer = setInterval(() => {
    if (stepIndex >= results[selected].steps.length - 1) { stopPlayback(); return; }
    renderStep(stepIndex + 1);
  }, Number($("playback-speed").value));
}

$("play-process").addEventListener("click", () => {
  if (playing) {
    stopPlayback();
    if (results[selected]?.steps.length) renderStep(stepIndex);
  } else startPlayback();
});
$("next-step").addEventListener("click", () => {
  stopPlayback();
  if (stepIndex < results[selected].steps.length - 1) renderStep(stepIndex + 1);
  else if (streaming) worker.postMessage({ type: "next", id: requestId });
});
$("restart-process").addEventListener("click", () => {
  stopPlayback();
  renderStep(0);
  startPlayback();
});
$("playback-speed").addEventListener("change", () => {
  if (playing) { stopPlayback(); startPlayback(); }
});

$("copy-name").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(titleCase(results[selected].name));
    $("copy-name").textContent = "已复制 ✓";
    setTimeout(() => $("copy-name").textContent = "复制名字 ↗", 1800);
  } catch { $("copy-name").textContent = "请选中名字后手动复制"; }
});

function exportImage() {
  const result = results[selected];
  const canvas = document.createElement("canvas");
  const width = 1440, cellWidth = 420, cellHeight = 260;
  canvas.width = width;
  canvas.height = 250 + Math.ceil(result.steps.length / 3) * cellHeight + 90;
  const context = canvas.getContext("2d");
  const text = (value, x, y, size = 16, color = "#1c2821", family = "'Microsoft YaHei', sans-serif") => {
    context.fillStyle = color;
    context.font = `${size}px ${family}`;
    context.fillText(value, x, y);
  };
  context.fillStyle = "#f5f4ed";
  context.fillRect(0, 0, width, canvas.height);
  text("name lab / 一个名字的诞生", 70, 65, 18, "#6b7b58");
  text(titleCase(result.name), 70, 140, 64, "#1c2821", "Georgia, serif");
  text(`${modes[result.mode].label}「${result.fragment}」 · ${result.steps.length} 个预测步骤 · ${result.terminated ? "以 EOS 结束" : "达到长度上限，未预测 EOS"}`, 72, 185, 17, "#758167");
  result.steps.forEach((step, index) => {
    const x = 70 + index % 3 * 450, y = 230 + Math.floor(index / 3) * cellHeight;
    context.fillStyle = "#fffefa";
    context.fillRect(x, y, cellWidth, cellHeight - 18);
    text(`STEP ${String(index + 1).padStart(2, "0")}`, x + 20, y + 27, 12, "#8b9878");
    const before = step.before || "开始";
    text(before.length > 19 ? before.slice(0, 18) + "…" : before, x + 20, y + 58, 20, "#1c2821", "Georgia, 'Microsoft YaHei', serif");
    text(`选中 ${letterLabel(step.char)} / ${percent(step.probability)}`, x + 20, y + 85, 13, "#5c7743");
    const max = Math.max(.1, step.top5[0].probability);
    step.top5.forEach((candidate, row) => {
      const top = y + 105 + row * 23;
      text(letterLabel(candidate.char), x + 20, top + 12, 13, "#5c6d4c");
      context.fillStyle = "#eff1e7";
      context.fillRect(x + 80, top, 240, 13);
      context.fillStyle = candidate.char === step.char ? "#577747" : "#c8d3b5";
      context.fillRect(x + 80, top, candidate.probability / max * 240, 13);
      text(percent(candidate.probability), x + 333, top + 12, 12, "#718061");
    });
  });
  text("Top-5 为原始模型概率；实际采样受随机性和补全约束影响。  ·  namelab", 72, canvas.height - 40, 14, "#8b967c");
  canvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `namelab-${result.name.replace(/[^a-z-]/gi, "") || "name"}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/png");
}
$("save-image").addEventListener("click", exportImage);

function renderLossChart() {
  if (!training) return;
  const key = $("training-model").value;
  const model = training.models[key];
  const history = model.history;
  const width = 560, height = 220, left = 38, right = 12, top = 12, bottom = 30;
  const values = history.flatMap(row => [row.train_loss, row.val_loss]);
  const min = Math.floor((Math.min(...values) - .025) * 10) / 10;
  const max = Math.ceil((Math.max(...values) + .025) * 10) / 10;
  const x = epoch => left + (epoch - 1) / Math.max(1, history.length - 1) * (width - left - right);
  const y = loss => top + (max - loss) / (max - min) * (height - top - bottom);
  const line = field => history.map(row => `${x(row.epoch).toFixed(2)},${y(row[field]).toFixed(2)}`).join(" ");
  const grid = Array.from({ length: 4 }, (_, i) => {
    const value = min + (max - min) / 3 * i;
    return `<line x1="${left}" x2="${width-right}" y1="${y(value)}" y2="${y(value)}" stroke="#dfe4d4" stroke-dasharray="3 5"/><text x="${left-9}" y="${y(value)+3}" text-anchor="end">${value.toFixed(2)}</text>`;
  }).join("");
  const ticks = [1, 10, 20, 30, 40, 50].filter(n => n <= history.length).map(n => `<text x="${x(n)}" y="${height-10}" text-anchor="middle">${n}</text>`).join("");
  const best = history.find(row => row.epoch === model.bestEpoch);
  $("loss-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="loss-title loss-desc" style="font:9px Consolas,monospace;fill:#8f9a80"><title id="loss-title">训练和验证损失曲线</title><desc id="loss-desc">共 ${history.length} 轮，最佳验证轮次 ${model.bestEpoch}，验证损失 ${model.bestValLoss.toFixed(4)}。</desc>${grid}${ticks}<polyline points="${line('train_loss')}" fill="none" stroke="#819b62" stroke-width="1.8"/><polyline points="${line('val_loss')}" fill="none" stroke="#374f3e" stroke-width="1.8"/><circle cx="${x(best.epoch)}" cy="${y(best.val_loss)}" r="4" fill="#dced90" stroke="#374f3e" stroke-width="1.5"/></svg>`;
  $("best-epoch").textContent = `最佳：第 ${model.bestEpoch} 轮 · ${model.bestValLoss.toFixed(4)}`;
}

fetch("./assets/training.json").then(response => {
  if (!response.ok) throw new Error("训练记录加载失败");
  return response.json();
}).then(data => { training = data; renderLossChart(); }).catch(() => {
  $("loss-chart").append(node("p", "error", "训练曲线暂时无法加载，请稍后刷新。"));
});
$("training-model").addEventListener("change", renderLossChart);
$("generator-form").addEventListener("submit", event => { event.preventDefault(); requestGeneration(true); });
$("temperature").addEventListener("input", () => $("temperature-value").textContent = Number($("temperature").value).toFixed(1));
document.querySelectorAll("[data-mode]").forEach(button => {
  button.addEventListener("click", () => updateMode(button.dataset.mode));
  button.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const keys = Object.keys(modes), current = keys.indexOf(mode);
    const index = event.key === "Home" ? 0 : event.key === "End" ? 2 : (current + (event.key === "ArrowRight" ? 1 : -1) + 3) % 3;
    updateMode(keys[index]);
    $(`mode-${keys[index]}`).focus();
  });
});
document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });
updateMode("prefix", false);
worker.postMessage({ type: "load" });
