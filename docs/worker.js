import { createModels, generateName, seededRandom } from "./engine.js?v=20261124-selection";

let models;
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
      const results = Array.from({ length: 6 }, () => generateName(models, data.options, random));
      self.postMessage({ type: "results", results, seed: data.seed, id: data.id });
    }
  } catch (error) {
    self.postMessage({ type: "error", message: error.message, id: data.id });
  }
};
