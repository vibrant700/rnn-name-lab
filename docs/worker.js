import { createModels, generateName, generateNameSteps, seededRandom } from "./engine.js";

let models;
let active;

function advance() {
  const next = active.iterator.next();
  if (next.done) {
    const results = [next.value, ...Array.from({ length: 5 }, () => generateName(models, active.options, active.random))];
    self.postMessage({ type: "results", results, seed: active.seed, id: active.id });
    active = undefined;
  } else {
    self.postMessage({ type: "step", step: next.value, id: active.id });
  }
}
self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const [manifest, binary] = await Promise.all([
        fetch("./assets/models.json").then(response => { if (!response.ok) throw new Error("模型加载失败"); return response.json(); }),
        fetch("./assets/weights.bin").then(response => { if (!response.ok) throw new Error("权重加载失败"); return response.arrayBuffer(); }),
      ]);
      models = createModels(manifest, binary);
      self.postMessage({ type: "ready" });
    } else if (data.type === "generate") {
      const random = seededRandom(data.seed);
      active = { id: data.id, seed: data.seed, options: data.options, random,
        iterator: generateNameSteps(models, data.options, random) };
      self.postMessage({ type: "started", options: data.options, id: data.id });
      advance();
    } else if (data.type === "next" && active?.id === data.id) {
      advance();
    }
  } catch (error) {
    self.postMessage({ type: "error", message: error.message, id: data.id });
  }
};
