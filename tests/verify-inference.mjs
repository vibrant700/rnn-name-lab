import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createModels, generateName, generateNameSteps, seededRandom } from "../docs/engine.js";

const manifest = JSON.parse(await readFile(new URL("../docs/assets/models.json", import.meta.url), "utf8"));
const binary = await readFile(new URL("../docs/assets/weights.bin", import.meta.url));
const models = createModels(manifest, binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength));
const fixtures = JSON.parse(await readFile(process.argv[2], "utf8"));
let maxError = 0, steps = 0;
for (const fixture of fixtures) {
  const model = models[fixture.model];
  let hidden = model.initialHidden();
  for (const reference of fixture.steps) {
    const actual = model.step(reference.char, hidden);
    hidden = actual.hidden;
    for (const key of ["hidden", "logits"]) {
      const error = Math.max(...reference[key].map((value, i) => Math.abs(value - actual[key][i])));
      maxError = Math.max(maxError, error);
      assert.ok(error < 0.002, `${fixture.model} ${key}: max absolute error ${error}`);
    }
    steps++;
  }
}
let cases = 0, terminated = 0;
for (const [mode, fragment] of [["prefix", "al"], ["suffix", "na"], ["middle", "ann"], ["middle", "a".repeat(18)]]) {
  for (const temperature of [.6, 1, 1.4]) {
    for (let seed = 0; seed < 30; seed++) {
      const options = { mode, fragment, temperature };
      const result = generateName(models, options, seededRandom(seed));
      assert.ok(result.name.length <= 20);
      assert.ok(!/[.^#]/.test(result.name));
      if (mode === "prefix") assert.ok(result.name.startsWith(fragment));
      if (mode === "suffix") assert.ok(result.name.endsWith(fragment));
      if (mode === "middle") assert.ok(result.name.slice(1, -1).includes(fragment));
      for (const step of result.steps) {
        assert.equal(step.distribution.length, mode === "middle" ? 32 : 30);
        assert.ok(Math.abs(step.distribution.reduce((sum, candidate) => sum + candidate.probability, 0) - 1) < 1e-12);
        assert.deepEqual(step.top5, [...step.distribution].sort((a, b) => b.probability - a.probability).slice(0, 5));
        assert.equal(step.top5.length, 5);
        assert.ok(step.top5.every((candidate, i, all) => i === 0 || candidate.probability <= all[i-1].probability));
      }
      if (result.terminated) {
        assert.equal(result.steps.at(-1).char, ".");
        terminated++;
      }
      assert.deepEqual(result, generateName(models, options, seededRandom(seed)));
      const streamed = generateNameSteps(models, options, seededRandom(seed));
      const liveSteps = [];
      let next = streamed.next();
      while (!next.done) { liveSteps.push(next.value); next = streamed.next(); }
      assert.deepEqual(liveSteps, result.steps);
      assert.deepEqual(next.value, result);
      cases++;
    }
  }
}
assert.ok(terminated > 0);
let greedyCases = 0;
for (const [mode, fragment] of [["prefix", "al"], ["suffix", "na"], ["middle", "ann"], ["middle", "a".repeat(18)]]) {
  const baseline = generateName(models, { mode, fragment, decoding: "greedy" }, () => { throw new Error("Greedy decoding must not sample randomness"); });
  for (const temperature of [.6, 1, 1.4]) {
    for (let seed = 0; seed < 10; seed++) {
      const result = generateName(models, { mode, fragment, temperature, decoding: "greedy" }, seededRandom(seed));
      assert.equal(result.name, baseline.name);
      assert.deepEqual(result.steps.map(step => step.char), baseline.steps.map(step => step.char));
      for (const step of result.steps) {
        let allowed = step.distribution;
        if (mode === "middle") {
          const separator = step.templateBefore.indexOf("#");
          const forceSeparator = separator < 0 && step.index - 1 === 20 - fragment.length - 1;
          allowed = allowed.filter(candidate => {
            if (forceSeparator) return candidate.char === "#";
            if (candidate.char === "^") return false;
            if (separator < 0) return candidate.char !== "." && (candidate.char !== "#" || step.templateBefore.length > 0);
            return candidate.char !== "#" && (candidate.char !== "." || step.templateBefore.length > separator + 1);
          });
        }
        const maximum = [...allowed].sort((a, b) => b.probability - a.probability)[0];
        assert.equal(step.char, maximum.char);
        assert.equal(step.samplingProbability, 1);
      }
      if (mode === "middle") assert.ok(result.name.slice(1, -1).includes(fragment));
      greedyCases++;
    }
  }
}
console.log(`PASS: ${steps} PyTorch/JavaScript steps; max error ${maxError.toExponential(3)}; ${cases} sampled cases (${terminated} EOS-terminated); ${greedyCases} greedy cases; greedy argmax over eligible characters, no RNG, invariant to temperature/seed.`);
