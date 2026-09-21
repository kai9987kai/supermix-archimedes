#!/usr/bin/env node
/**
 * Headless Fly Lab runner.
 *
 * Replicates the step() loop in DIAMOND-SIM-FLY/fly-diamond-nexus.html exactly
 * (environment tick, chemical fields, bilateral antennae, 22-brain syncytium,
 * three agents, reinforcement) with the drawing removed, runs it for a wall-clock
 * budget, and writes the v8 FlyBrainStateSerializer snapshot plus a telemetry
 * trace to disk. That snapshot is what the Python distillation stage consumes.
 *
 *   node fly_lab_headless.cjs --minutes 10 --seed 2077 --out fly_run
 */
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const ENGINE = path.join(__dirname, "fly-diamond-nexus", "DIAMOND-SIM-FLY", "src");
const {
  TwentyTwoFlyBrainSyncytium, MultiAgentGraphGraft, ChemicalFieldGrid,
  MulberryPRNG, FlyBrainStateSerializer, PreTrainingEngine
} = require(path.join(ENGINE, "fly-brain-engine.js"));
const { FlyEnvironment } = require(path.join(ENGINE, "fly-environment.js"));

// Constants copied from fly-diamond-nexus.html so the headless world matches.
const GRID = 20;
const DIAMOND_COUNT = 12;
const HAZARD_COUNT = 8;
const INIT_ENERGY = 120;
const ENERGY_STEP = 1;
const ENERGY_DIAMOND = 25;
const ENERGY_HAZARD = 30;
const ENERGY_MAX = 180;
const DIAMOND_SENSE = 6;
const HAZARD_SENSE = 4;

function parseArgs(argv) {
  const o = { minutes: 10, seed: 2077, out: "fly_run", coupling: 0.8, pretrainEpochs: 15, weatherCycle: 1500, respawnEnergy: true };
  for (let i = 0; i < argv.length; i++) {
    const f = argv[i], v = argv[i + 1];
    switch (f) {
      case "--minutes": o.minutes = Number(v); i++; break;
      case "--seed": o.seed = Number(v); i++; break;
      case "--out": o.out = v; i++; break;
      case "--coupling": o.coupling = Number(v); i++; break;
      case "--pretrain": o.pretrainEpochs = Number(v); i++; break;
      case "--weather-cycle": o.weatherCycle = Number(v); i++; break;
      default: throw new Error("unknown flag " + f);
    }
  }
  return o;
}

function generateWorld(seed) {
  const rng = new MulberryPRNG(seed);
  const diamonds = [], hazards = [];
  const occupied = new Set();
  const key = (x, y) => `${x},${y}`;
  let a1x, a1y, a2x, a2y, a3x, a3y;
  do { a1x = rng.int(GRID); a1y = rng.int(GRID); } while (occupied.has(key(a1x, a1y)));
  occupied.add(key(a1x, a1y));
  do { a2x = rng.int(GRID); a2y = rng.int(GRID); } while (occupied.has(key(a2x, a2y)));
  occupied.add(key(a2x, a2y));
  do { a3x = rng.int(GRID); a3y = rng.int(GRID); } while (occupied.has(key(a3x, a3y)));
  occupied.add(key(a3x, a3y));
  while (diamonds.length < DIAMOND_COUNT) {
    const x = rng.int(GRID), y = rng.int(GRID);
    if (!occupied.has(key(x, y))) { diamonds.push({ x, y }); occupied.add(key(x, y)); }
  }
  while (hazards.length < HAZARD_COUNT) {
    const x = rng.int(GRID), y = rng.int(GRID);
    if (!occupied.has(key(x, y))) { hazards.push({ x, y }); occupied.add(key(x, y)); }
  }
  return {
    agent1X: a1x, agent1Y: a1y, agent1Energy: INIT_ENERGY,
    agent2X: a2x, agent2Y: a2y, agent2Energy: INIT_ENERGY,
    agent3X: a3x, agent3Y: a3y, agent3Energy: INIT_ENERGY,
    diamonds, hazards, diamondsCollected: 0, hazardsHit: 0, seed, gridSize: GRID
  };
}

class HeadlessFlyLab {
  constructor(seed, coupling) {
    this.seed = seed;
    this.syncytium = new TwentyTwoFlyBrainSyncytium(seed, { couplingStrength: coupling });
    this.graft = new MultiAgentGraphGraft(this.syncytium, 0.55);
    this.chemGrid = new ChemicalFieldGrid(GRID, GRID);
    this.climate = new FlyEnvironment(seed, GRID);
    this.env = generateWorld(seed);
    this.climate.seedTerrain(this.env);
    this.graft.beaconWaypoints = [];
    this.simStep = 0;
    this.agentDiamonds = { agent1: 0, agent2: 0, agent3: 0 };
    this.agentHazards = { agent1: 0, agent2: 0, agent3: 0 };
    this.agentPredator = { agent1: 0, agent2: 0, agent3: 0 };
    this.respawns = 0;
    this.events = { stasis: 0, gfEscape: 0, handshake: 0, triSwarm: 0, beacons: 0 };
    this.lastResult = null;
    // Experience log: (obs, action, reward, ego, brain descending outputs) per agent per tick.
    // Sampled, not every tick, to keep the file bounded.
    this.experience = [];
  }

  buildSensoryObs(agentX, agentY, agentEnergy, otherX, otherY, heading = 0) {
    const env = this.env, climate = this.climate, chemGrid = this.chemGrid;
    const dnX = agentX / GRID, dnY = agentY / GRID;
    const normE = agentEnergy / INIT_ENERGY;
    let nearestDiamond = 999, nearestHazard = 999, diamondAngle = 0, hazardAngle = 0;
    for (const d of env.diamonds) {
      const dist = Math.hypot(d.x - agentX, d.y - agentY);
      if (dist < nearestDiamond) { nearestDiamond = dist; diamondAngle = Math.atan2(d.y - agentY, d.x - agentX); }
    }
    for (const h of [...env.hazards, ...climate.predators]) {
      const dist = Math.hypot(h.x - agentX, h.y - agentY);
      if (dist < nearestHazard) { nearestHazard = dist; hazardAngle = Math.atan2(h.y - agentY, h.x - agentX); }
    }
    const weather = climate.sense(agentX, agentY, env);
    const scentGain = Math.max(0.45, Math.min(1.3, 0.8 + weather.humidity * 0.4 - weather.precipitation * 0.2 +
      (Math.cos(diamondAngle) * weather.windX + Math.sin(diamondAngle) * weather.windY) * -0.12));
    const diamondReward = nearestDiamond < DIAMOND_SENSE ? Math.min(1, (DIAMOND_SENSE - nearestDiamond) / DIAMOND_SENSE * scentGain) : 0;
    const hazardThreat = nearestHazard < HAZARD_SENSE ? (HAZARD_SENSE - nearestHazard) / HAZARD_SENSE : 0;
    const opticFlowX = (otherX - agentX) / GRID;
    const opticFlowY = (otherY - agentY) / GRID;
    const biFood = chemGrid.sampleBilateral("foodOdor", agentX, agentY, heading, 0.8);
    const biCustom = chemGrid.sampleBilateral("customPheromone", agentX, agentY, heading, 0.8);
    const leftAntenna = Math.min(1.0, biFood.left + biCustom.left);
    const rightAntenna = Math.min(1.0, biFood.right + biCustom.right);
    const observation = [
      dnX, dnY, diamondReward, hazardThreat, normE,
      Math.min(nearestDiamond / GRID, 1),
      Math.sin(diamondAngle), Math.cos(diamondAngle),
      Math.sin(hazardAngle), Math.cos(hazardAngle),
      opticFlowX * weather.visibility, opticFlowY * weather.visibility,
      Math.min(1, leftAntenna * scentGain), Math.min(1, rightAntenna * scentGain)
    ];
    observation.environment = weather;
    return observation;
  }

  applyAction(agentKey, actionIndex) {
    const env = this.env, climate = this.climate;
    const dx = [0, 0, -1, 1][actionIndex], dy = [-1, 1, 0, 0][actionIndex];
    const xKey = agentKey + "X", yKey = agentKey + "Y";
    const nx = env[xKey] + dx, ny = env[yKey] + dy;
    if (!climate.isBlocked(nx, ny)) { env[xKey] = nx; env[yKey] = ny; }
    env[agentKey + "MovementCost"] = climate.movementCost(env[xKey], env[yKey], dx, dy);
  }

  checkInteractions(agentKey) {
    const env = this.env, climate = this.climate, graft = this.graft;
    const ax = env[agentKey + "X"], ay = env[agentKey + "Y"];
    const eKey = agentKey + "Energy";
    let reward = 0;
    env[eKey] = Math.max(0, env[eKey] - (env[agentKey + "MovementCost"] ?? ENERGY_STEP));
    const predatorDamage = climate.predatorDamage(ax, ay);
    if (predatorDamage > 0) {
      env[eKey] = Math.max(0, env[eKey] - predatorDamage);
      env.hazardsHit++;
      this.agentPredator[agentKey]++;
      graft.feedback(-3, true, { x: ax / GRID, y: ay / GRID }, agentKey);
      reward -= 3;
    }
    for (let i = env.diamonds.length - 1; i >= 0; i--) {
      if (env.diamonds[i].x === ax && env.diamonds[i].y === ay) {
        const gain = ENERGY_DIAMOND * (env.diamonds[i].strength || 1);
        env.diamonds.splice(i, 1);
        env[eKey] = Math.min(ENERGY_MAX, env[eKey] + gain);
        env.diamondsCollected++;
        this.agentDiamonds[agentKey]++;
        graft.feedback(5, false, { x: ax / GRID, y: ay / GRID }, agentKey);
        reward += 5;
        break;
      }
    }
    for (const h of env.hazards) {
      if (h.x === ax && h.y === ay) {
        env[eKey] = Math.max(0, env[eKey] - ENERGY_HAZARD * (h.strength || 1));
        env.hazardsHit++;
        this.agentHazards[agentKey]++;
        graft.feedback(-3, true, { x: ax / GRID, y: ay / GRID }, agentKey);
        reward -= 3;
        break;
      }
    }
    // The browser app leaves a fly at zero energy stranded; for a long
    // training run we top it back up so all three agents keep generating
    // experience. Counted so the receipt is honest about it.
    if (env[eKey] <= 0) { env[eKey] = INIT_ENERGY * 0.5; this.respawns++; }
    return reward;
  }

  step() {
    const env = this.env, climate = this.climate, chemGrid = this.chemGrid, graft = this.graft;
    this.simStep++;
    climate.step(env);
    for (const d of env.diamonds) chemGrid.emit("foodOdor", d.x, d.y, 0.4);
    for (const h of env.hazards) chemGrid.emit("threatOdor", h.x, h.y, 0.4);
    chemGrid.emit("foragerTrail", env.agent1X, env.agent1Y, 0.6);
    chemGrid.emit("sentinelTrail", env.agent2X, env.agent2Y, 0.6);
    chemGrid.emit("customPheromone", env.agent3X, env.agent3Y, 0.35);
    chemGrid.step(0.16 + climate.weather.humidity * 0.08, 0.96 - climate.weather.precipitation * 0.06);

    const obs1 = this.buildSensoryObs(env.agent1X, env.agent1Y, env.agent1Energy, env.agent2X, env.agent2Y, graft.agent1.lastHeading);
    const obs2 = this.buildSensoryObs(env.agent2X, env.agent2Y, env.agent2Energy, env.agent1X, env.agent1Y, graft.agent2.lastHeading);
    const obs3 = this.buildSensoryObs(env.agent3X, env.agent3Y, env.agent3Energy, env.agent1X, env.agent1Y, graft.agent3.lastHeading);
    env.environmentByAgent = { agent1: obs1.environment, agent2: obs2.environment, agent3: obs3.environment };

    const beaconsBefore = graft.beaconWaypoints.length;
    const result = graft.resolveAgents(env, obs1, obs2, obs3);
    this.lastResult = result;

    this.applyAction("agent1", result.agent1.actionIndex);
    this.applyAction("agent2", result.agent2.actionIndex);
    if (result.agent3) this.applyAction("agent3", result.agent3.actionIndex);

    for (const k of ["agent1", "agent2", "agent3"]) {
      const s = result[k];
      if (!s) continue;
      if (s.isStasis) this.events.stasis++;
      if (s.isGFEscape) this.events.gfEscape++;
    }
    if (result.isAcousticHandshake) this.events.handshake++;
    if (result.isTriSwarmResonance) this.events.triSwarm++;
    if (graft.beaconWaypoints.length > beaconsBefore) this.events.beacons++;

    const r1 = this.checkInteractions("agent1");
    const r2 = this.checkInteractions("agent2");
    const r3 = this.checkInteractions("agent3");

    return { obs: [obs1, obs2, obs3], result, rewards: [r1, r2, r3] };
  }

  /** Per-brain descending outputs + neuromodulators, the teacher signal for distillation. */
  brainReadout() {
    return this.syncytium.brains.map(b => ({
      role: b.role,
      desc: Array.from(b.descendingOutputs),
      pam: b.dopaminePAM, ppl1: b.dopaminePPL1, oa: b.octopamineOA, ht: b.serotonin5HT,
      sat: b.metabolicSatiety, npf: b.neuropeptideNPF, sif: b.neuropeptideSIFamide,
      ring: b.ringCertainty, spars: b.kcSparsity, apl: b.aplActivity
    }));
  }

  telemetry() {
    const t = this.syncytium.getTelemetry();
    return {
      step: this.simStep, tick: this.syncytium.tickCount,
      diamonds: this.env.diamondsCollected, hazards: this.env.hazardsHit,
      perAgent: { diamonds: { ...this.agentDiamonds }, hazards: { ...this.agentHazards }, predator: { ...this.agentPredator } },
      energy: [this.env.agent1Energy, this.env.agent2Energy, this.env.agent3Energy],
      regimes: [this.graft.agent1.behavioralRegime, this.graft.agent2.behavioralRegime, this.graft.agent3.behavioralRegime],
      mitosis: this.syncytium.neurogenesis.mitosisCount, apoptosis: this.syncytium.neurogenesis.apoptosisCount,
      bornTotal: this.syncytium.brains.reduce((a, b) => a + b.bornNeurons.length, 0),
      lastRPE: this.syncytium.lastRPE, engrams: this.syncytium.engramBank.length,
      weather: { ...this.climate.weather }, preset: this.climate.preset,
      events: { ...this.events }, respawns: this.respawns,
      graph: t.graph, specialists: t.specialists,
      predators: this.climate.predators.length, resourcesGrown: this.climate.resourcesGrown
    };
  }
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(opts.out);
  fs.mkdirSync(outDir, { recursive: true });
  const budgetMs = opts.minutes * 60 * 1000;

  console.log(`[fly-lab] seed=${opts.seed} coupling=${opts.coupling} budget=${opts.minutes}min out=${outDir}`);
  const lab = new HeadlessFlyLab(opts.seed, opts.coupling);
  console.log(`[fly-lab] ${lab.syncytium.brainCount} brains: ${lab.syncytium.roles.join(", ")}`);
  console.log(`[fly-lab] commissural matrix ${lab.syncytium.brainCount}x${lab.syncytium.brainCount}x4 = ${lab.syncytium.commissuralWeights.length} slots`);

  // Same offline conditioning the app's Pre-Train button runs.
  if (opts.pretrainEpochs > 0) {
    PreTrainingEngine.runPreTraining(lab.syncytium, opts.pretrainEpochs);
    console.log(`[fly-lab] pre-trained ${opts.pretrainEpochs} epochs`);
  }

  const presets = ["temperate", "rain", "drought", "storm", "calm"];
  const traceStream = fs.createWriteStream(path.join(outDir, "telemetry.jsonl"));
  const expStream = fs.createWriteStream(path.join(outDir, "experience.jsonl"));
  const t0 = Date.now();
  let lastLog = t0, lastSnapshot = t0;
  let expRows = 0;

  const snapshot = (label) => {
    const json = FlyBrainStateSerializer.serialize(lab.graft, {
      headless: true, label, simStep: lab.simStep, seed: opts.seed,
      env: { agent1: [lab.env.agent1X, lab.env.agent1Y, lab.env.agent1Energy],
             agent2: [lab.env.agent2X, lab.env.agent2Y, lab.env.agent2Energy],
             agent3: [lab.env.agent3X, lab.env.agent3Y, lab.env.agent3Energy],
             diamonds: lab.env.diamonds, hazards: lab.env.hazards },
      climate: lab.climate.snapshot(),
      telemetry: lab.telemetry()
    });
    fs.writeFileSync(path.join(outDir, `brain_state_${label}.json`), json);
    return json.length;
  };

  snapshot("initial");

  while (Date.now() - t0 < budgetMs) {
    // Rotate weather every N ticks so the climate/plume/threat specialists see every regime.
    if (opts.weatherCycle > 0 && lab.simStep > 0 && lab.simStep % opts.weatherCycle === 0) {
      const p = presets[Math.floor(lab.simStep / opts.weatherCycle) % presets.length];
      lab.climate.setPreset(p);
    }
    const { obs, result, rewards } = lab.step();

    // Every 4th tick, log an experience row per agent: what it saw, what the 22
    // brains output, what it did, what it got. This is the distillation corpus.
    if (lab.simStep % 6 === 0) {
      const brains = lab.brainReadout();
      const keys = ["agent1", "agent2", "agent3"];
      for (let i = 0; i < 3; i++) {
        const s = result[keys[i]];
        if (!s) continue;
        expStream.write(JSON.stringify({
          step: lab.simStep, agent: keys[i], regime: lab.graft[keys[i]].behavioralRegime,
          obs: obs[i].map(v => +v.toFixed(5)),
          env: { t: +obs[i].environment.temperature.toFixed(2), h: +obs[i].environment.humidity.toFixed(3),
                 wx: +obs[i].environment.windX.toFixed(3), wy: +obs[i].environment.windY.toFixed(3),
                 rain: +obs[i].environment.precipitation.toFixed(3), pred: +obs[i].environment.predatorProximity.toFixed(3),
                 refuge: obs[i].environment.inRefuge ? 1 : 0, water: obs[i].environment.inWater ? 1 : 0 },
          probs: (s.probabilities || s.probs || []).map(v => +v.toFixed(5)),
          action: s.actionIndex, reward: rewards[i],
          stasis: s.isStasis ? 1 : 0, gf: s.isGFEscape ? 1 : 0,
          brains: brains.map(b => ({ r: b.role, d: b.desc.map(v => +v.toFixed(4)), pam: +b.pam.toFixed(3), ppl1: +b.ppl1.toFixed(3) }))
        }) + "\n");
        expRows++;
      }
    }

    const now = Date.now();
    if (now - lastLog >= 15000) {
      const tel = lab.telemetry();
      traceStream.write(JSON.stringify({ t: now - t0, ...tel }) + "\n");
      const pct = ((now - t0) / budgetMs * 100).toFixed(1);
      console.log(`[fly-lab] ${pct}% step=${tel.step} diamonds=${tel.diamonds} hazards=${tel.hazards} born=${tel.bornTotal} mitosis=${tel.mitosis} apoptosis=${tel.apoptosis} engrams=${tel.engrams} rpe=${tel.lastRPE.toFixed(3)} weather=${tel.preset} regimes=${tel.regimes.join("/")} exp=${expRows}`);
      lastLog = now;
    }
    if (now - lastSnapshot >= 120000) {
      snapshot(`t${Math.round((now - t0) / 60000)}m`);
      lastSnapshot = now;
    }
  }

  const finalTel = lab.telemetry();
  traceStream.write(JSON.stringify({ t: Date.now() - t0, ...finalTel, final: true }) + "\n");
  traceStream.end();
  expStream.end();
  const bytes = snapshot("final");
  const receipt = {
    seed: opts.seed, coupling: opts.coupling, minutes: opts.minutes, wallMs: Date.now() - t0,
    steps: lab.simStep, experienceRows: expRows, finalSnapshotBytes: bytes,
    brainCount: lab.syncytium.brainCount, roles: lab.syncytium.roles,
    telemetry: finalTel,
    engineHash: require("node:crypto").createHash("sha256").update(fs.readFileSync(path.join(ENGINE, "fly-brain-engine.js"))).digest("hex")
  };
  fs.writeFileSync(path.join(outDir, "receipt.json"), JSON.stringify(receipt, null, 2));
  console.log(`[fly-lab] DONE steps=${lab.simStep} wall=${((Date.now() - t0) / 1000).toFixed(1)}s experience=${expRows} rows`);
  console.log(`[fly-lab] final: diamonds=${finalTel.diamonds} hazards=${finalTel.hazards} born=${finalTel.bornTotal} mitosis=${finalTel.mitosis} apoptosis=${finalTel.apoptosis} engrams=${finalTel.engrams}`);
  console.log(`[fly-lab] per-agent diamonds: ${JSON.stringify(finalTel.perAgent.diamonds)} respawns=${lab.respawns}`);
}

main();
