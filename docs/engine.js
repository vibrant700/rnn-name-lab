// 与 names_rnn.RNN 相同：输出使用旧 hidden，新 hidden 经 tanh 更新。
export class RNN {
  constructor(meta, weights) {
    this.chars = meta.chars;
    this.hiddenSize = meta.hiddenSize;
    this.size = this.chars.length;
    this.indices = new Map(this.chars.map((c, i) => [c, i]));
    this.tensors = {};
    for (const [name, { offset, shape }] of Object.entries(meta.tensors)) {
      this.tensors[name] = weights.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
    }
  }

  step(char, hidden) {
    const index = this.indices.get(char);
    if (index === undefined) throw new Error(`模型不认识字符「${char}」`);
    const width = this.size + this.hiddenSize;
    const linear = (name, rows) => {
      const matrix = this.tensors[`${name}.weight`];
      const bias = this.tensors[`${name}.bias`];
      const values = new Float64Array(rows);
      for (let row = 0; row < rows; row++) {
        const start = row * width;
        let value = bias[row] + matrix[start + index];
        for (let h = 0; h < this.hiddenSize; h++) value += matrix[start + this.size + h] * hidden[h];
        values[row] = value;
      }
      return values;
    };
    return { logits: linear("i2o", this.size), hidden: linear("i2h", this.hiddenSize).map(Math.tanh) };
  }

  initialHidden() { return new Float64Array(this.hiddenSize); }
}

export function createModels(manifest, buffer) {
  const weights = new Float32Array(buffer);
  return Object.fromEntries(Object.entries(manifest.models).map(([name, meta]) => [name, new RNN(meta, weights)]));
}

export function softmax(logits, temperature = 1) {
  const max = Math.max(...logits);
  const probabilities = Array.from(logits, value => Math.exp((value - max) / temperature));
  const sum = probabilities.reduce((a, b) => a + b, 0);
  return probabilities.map(value => value / sum);
}

export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function sample(probabilities, random) {
  let remaining = random();
  for (let i = 0; i < probabilities.length; i++) {
    remaining -= probabilities[i];
    if (remaining < 0) return i;
  }
  return probabilities.length - 1;
}

function recordStep(model, logits, samplingLogits, random, temperature, before, after) {
  const original = softmax(logits);
  const sampling = softmax(samplingLogits, temperature);
  const selected = sample(sampling, random);
  const distribution = original.map((probability, index) => ({ char: model.chars[index], probability }));
  const top5 = [...distribution].sort((a, b) => b.probability - a.probability).slice(0, 5);
  return {
    before, after: after(model.chars[selected]), char: model.chars[selected], top5, distribution,
    probability: original[selected], samplingProbability: sampling[selected],
    constrained: Array.from(logits).some((value, i) => value !== samplingLogits[i]),
  };
}

export function* generateNameSteps(models, { mode, fragment, temperature = 1, maxLength = 20 }, random = Math.random) {
  const model = models[mode === "suffix" ? "backward" : mode === "middle" ? "decoder" : "forward"];
  if (!fragment.trim() || [...fragment].some(c => !model.indices.has(c) || ".^#".includes(c))) {
    throw new Error("请输入英文名字片段，可包含字母、空格、连字符或撇号。");
  }
  if (fragment.length > maxLength - (mode === "middle" ? 2 : 1)) throw new Error("片段太长，请留出补全字母的位置。");
  const steps = [];
  let hidden = model.initialHidden();
  let template = "";
  let logits;
  const reverse = text => [...text].reverse().join("");
  const display = value => mode === "suffix" ? reverse(value) : value.replace("#", fragment);
  let limit;
  if (mode === "middle") {
    for (const c of fragment) hidden = models.encoder.step(c, hidden).hidden;
    limit = maxLength - fragment.length + 1;
  } else {
    template = mode === "suffix" ? reverse(fragment) : fragment;
    for (const c of template) ({ hidden, logits } = model.step(c, hidden));
    limit = maxLength - fragment.length + 1; // 最后一步仍允许预测 EOS。
  }
  let previous = "^";
  let terminated = false;
  let truncated = false;
  for (let i = 0; i < limit; i++) {
    if (mode === "middle") ({ hidden, logits } = model.step(previous, hidden));
    const allowed = Array.from(logits);
    if (mode === "middle") {
      allowed[model.indices.get("^")] = -Infinity;
      const separator = template.indexOf("#");
      if (separator < 0) {
        allowed[model.indices.get(".")] = -Infinity;
        if (!template) allowed[model.indices.get("#")] = -Infinity;
        if (i === limit - 2) {
          allowed.fill(-Infinity);
          allowed[model.indices.get("#")] = 0;
        }
      } else {
        allowed[model.indices.get("#")] = -Infinity;
        if (template.length === separator + 1) allowed[model.indices.get(".")] = -Infinity;
      }
    }
    const step = recordStep(model, logits, allowed, random, temperature, display(template),
      char => char === "." ? display(template) : display(template + char));
    if (mode !== "middle" && template.length >= maxLength && step.char !== ".") {
      // 长度上限不伪装成模型预测的 EOS。
      truncated = true;
      break;
    }
    steps.push({ ...step, index: steps.length + 1, templateBefore: template });
    yield steps.at(-1);
    if (step.char === ".") { terminated = true; break; }
    template += step.char;
    previous = step.char;
    if (mode !== "middle") ({ hidden, logits } = model.step(step.char, hidden));
  }
  if (!terminated) truncated = true;
  return { name: display(template), mode, fragment, steps, terminated, truncated, temperature };
}

export function generateName(models, options, random = Math.random) {
  const sequence = generateNameSteps(models, options, random);
  let next = sequence.next();
  while (!next.done) next = sequence.next();
  return next.value;
}
