/**
 * MoE expert routing failure planner.
 *
 * Pure, framework-agnostic planner: given token routes, expert capacities,
 * worker health, already-received tokens, gradient versions and a retry
 * window, it decides for every affected token whether to WAIT for the
 * worker, REROUTE the token to another expert, or mark it FAILED.
 *
 * Safety invariants enforced by the planner:
 *  - A token is never assigned to two experts for the same routing slot.
 *  - A rerouted token never pushes an expert beyond its batch capacity.
 *  - Tokens already computed at the current gradient version are untouched.
 */

export const Action = Object.freeze({
  WAIT: 'wait',
  REROUTE: 'reroute',
  FAIL: 'fail',
});

/**
 * @typedef {Object} TokenRoute
 * @property {string} expert  expert id
 * @property {number} slot    top-k slot (0..k-1) this route belongs to
 * @property {number} [weight] router weight for this slot
 *
 * @typedef {Object} PlanInput
 * @property {Array<{id: string, routes: TokenRoute[]}>} tokens
 * @property {Object<string, number>} capacities  expertId -> max tokens per batch
 * @property {Object<string, {status: 'alive'|'dead', experts: string[], deadSince?: number}>} workers
 * @property {Object<string, string>} expertToWorker  expertId -> workerId
 * @property {Array<{tokenId: string, expert: string, slot: number,
 *   status: 'queued'|'computing'|'done', gradientVersion: number}>} received
 * @property {number} currentGradientVersion
 * @property {{now: number, maxWaitMs: number}} retryWindow
 */

/**
 * Build the recovery plan.
 *
 * @param {PlanInput} input
 * @returns {{
 *   decisions: Array<{tokenId: string, expert: string, slot: number, action: string, reason: string}>,
 *   reroutes: Array<{tokenId: string, slot: number, fromExpert: string, toExpert: string}>,
 *   waits: Array<{tokenId: string, expert: string, slot: number, waitMs: number}>,
 *   rollbackSteps: number[],
 *   unrecoverable: Array<{tokenId: string, slot: number, expert: string, reason: string}>,
 *   plannedLoad: Object<string, number>
 * }}
 */
export function planRecovery(input) {
  const {
    tokens,
    capacities,
    workers,
    expertToWorker,
    received,
    currentGradientVersion,
    retryWindow,
  } = input;

  const decisions = [];
  const reroutes = [];
  const waits = [];
  const unrecoverable = [];
  const rollbackSet = new Set();

  // Index received records by token+slot so we can see what each expert
  // already has. Duplicate deliveries for the same slot are a routing bug
  // and must be rolled back, never silently kept.
  const receivedByKey = new Map();
  for (const rec of received) {
    const key = `${rec.tokenId}${rec.slot}`;
    if (!receivedByKey.has(key)) receivedByKey.set(key, []);
    receivedByKey.get(key).push(rec);
  }

  // Baseline load per expert from healthy, current-version deliveries.
  const plannedLoad = {};
  for (const expert of Object.keys(capacities)) plannedLoad[expert] = 0;
  for (const rec of received) {
    if (rec.status === 'done' && rec.gradientVersion === currentGradientVersion) {
      plannedLoad[rec.expert] = (plannedLoad[rec.expert] ?? 0) + 1;
    }
  }

  const workerOf = (expert) => expertToWorker[expert];
  const workerAlive = (expert) => {
    const w = workers[workerOf(expert)];
    return w !== undefined && w.status === 'alive';
  };
  const remainingWaitMs = (expert) => {
    const w = workers[workerOf(expert)];
    if (!w || w.status !== 'dead' || w.deadSince === undefined) return 0;
    return retryWindow.maxWaitMs - (retryWindow.now - w.deadSince);
  };

  // Experts that lost their worker and will not be waited for.
  const isFailedExpert = (expert) => !workerAlive(expert) && remainingWaitMs(expert) <= 0;

  for (const token of tokens) {
    const seenSlots = new Set();
    for (const route of token.routes) {
      // Route-table duplicate: the same slot appears twice, which would
      // send one token to two experts for a single slot.
      if (seenSlots.has(route.slot)) {
        rollbackSet.add(currentGradientVersion);
        decisions.push({
          tokenId: token.id,
          expert: route.expert,
          slot: route.slot,
          action: Action.FAIL,
          reason: 'duplicate-route-slot',
        });
        continue;
      }
      seenSlots.add(route.slot);

      const key = `${token.id}${route.slot}`;
      const deliveries = receivedByKey.get(key) ?? [];

      // --- duplicate routing detection ---------------------------------
      const doneCurrent = deliveries.filter(
        (d) => d.status === 'done' && d.gradientVersion === currentGradientVersion,
      );
      if (doneCurrent.length > 1) {
        // Same token+slot computed by two experts: keep the first, roll
        // back every extra delivery's gradient contribution.
        for (const extra of doneCurrent.slice(1)) {
          rollbackSet.add(extra.gradientVersion);
          decisions.push({
            tokenId: token.id,
            expert: extra.expert,
            slot: route.slot,
            action: Action.FAIL,
            reason: 'duplicate-route',
          });
        }
        continue;
      }
      if (doneCurrent.length === 1) {
        // Already computed at the current version: nothing to do.
        continue;
      }

      // Stale completed work contributes gradients from an old version.
      for (const d of deliveries) {
        if (d.status === 'done' && d.gradientVersion < currentGradientVersion) {
          rollbackSet.add(d.gradientVersion);
        }
      }

      const pending = deliveries.find((d) => d.status !== 'done');
      const expert = pending ? pending.expert : route.expert;

      // --- healthy path -------------------------------------------------
      if (workerAlive(expert)) {
        if (!pending) {
          // Dispatched in the route table but never delivered; count the
          // in-flight load so reroutes do not oversubscribe this expert.
          plannedLoad[expert] = (plannedLoad[expert] ?? 0) + 1;
        }
        continue;
      }

      // --- wait path ----------------------------------------------------
      const waitMs = remainingWaitMs(expert);
      if (waitMs > 0) {
        waits.push({ tokenId: token.id, expert, slot: route.slot, waitMs });
        decisions.push({
          tokenId: token.id,
          expert,
          slot: route.slot,
          action: Action.WAIT,
          reason: 'worker-retry-window-open',
        });
        // Reserve capacity: the token is still expected to land here.
        plannedLoad[expert] = (plannedLoad[expert] ?? 0) + 1;
        continue;
      }

      // --- reroute path ---------------------------------------------------
      // Candidates: the token's other top-k experts first (router already
      // scored them), then any alive expert. Never an expert that already
      // holds this token+slot, never a failed expert, never over capacity.
      const candidates = [];
      const seen = new Set([expert]);
      for (const alt of token.routes) {
        if (alt.slot === route.slot || seen.has(alt.expert)) continue;
        seen.add(alt.expert);
        candidates.push(alt.expert);
      }
      for (const altExpert of Object.keys(capacities)) {
        if (!seen.has(altExpert)) {
          seen.add(altExpert);
          candidates.push(altExpert);
        }
      }

      let target = null;
      for (const cand of candidates) {
        if (isFailedExpert(cand) || !workerAlive(cand)) continue;
        const load = plannedLoad[cand] ?? 0;
        const cap = capacities[cand] ?? 0;
        if (load + 1 <= cap) {
          target = cand;
          break;
        }
      }

      if (target === null) {
        unrecoverable.push({
          tokenId: token.id,
          slot: route.slot,
          expert,
          reason: 'no-alive-expert-with-capacity',
        });
        decisions.push({
          tokenId: token.id,
          expert,
          slot: route.slot,
          action: Action.FAIL,
          reason: 'no-alive-expert-with-capacity',
        });
        continue;
      }

      plannedLoad[target] += 1;
      reroutes.push({
        tokenId: token.id,
        slot: route.slot,
        fromExpert: expert,
        toExpert: target,
      });
      decisions.push({
        tokenId: token.id,
        expert,
        slot: route.slot,
        action: Action.REROUTE,
        reason: 'worker-dead-past-retry-window',
      });
    }
  }

  const rollbackSteps = [...rollbackSet].sort((a, b) => a - b);
  return { decisions, reroutes, waits, rollbackSteps, unrecoverable, plannedLoad };
}
