import { test } from 'node:test';
import assert from 'node:assert/strict';
import { planRecovery, Action } from '../src/planner.js';

const baseInput = () => ({
  tokens: [],
  capacities: {},
  workers: {},
  expertToWorker: {},
  received: [],
  currentGradientVersion: 5,
  retryWindow: { now: 1000, maxWaitMs: 500 },
});

test('single worker loss: queued tokens reroute to alive expert with capacity', () => {
  const input = baseInput();
  input.tokens = [
    { id: 't1', routes: [{ expert: 'e1', slot: 0 }, { expert: 'e2', slot: 1 }] },
    { id: 't2', routes: [{ expert: 'e1', slot: 0 }, { expert: 'e3', slot: 1 }] },
  ];
  input.capacities = { e1: 2, e2: 2, e3: 2 };
  input.workers = {
    w1: { status: 'dead', experts: ['e1'], deadSince: 100 }, // past retry window
    w2: { status: 'alive', experts: ['e2', 'e3'] },
  };
  input.expertToWorker = { e1: 'w1', e2: 'w2', e3: 'w2' };
  input.received = [
    { tokenId: 't1', expert: 'e1', slot: 0, status: 'queued', gradientVersion: 5 },
    { tokenId: 't2', expert: 'e1', slot: 0, status: 'computing', gradientVersion: 5 },
  ];

  const plan = planRecovery(input);
  assert.equal(plan.reroutes.length, 2);
  assert.deepEqual(
    plan.reroutes.map((r) => [r.tokenId, r.fromExpert, r.toExpert]),
    [
      ['t1', 'e1', 'e2'],
      ['t2', 'e1', 'e3'],
    ],
  );
  assert.equal(plan.unrecoverable.length, 0);
  assert.equal(plan.waits.length, 0);
  // Capacity respected: no expert exceeds its limit.
  for (const [expert, load] of Object.entries(plan.plannedLoad)) {
    assert.ok(load <= input.capacities[expert], `${expert} overloaded`);
  }
});

test('single worker loss inside retry window: planner waits instead of rerouting', () => {
  const input = baseInput();
  input.tokens = [{ id: 't1', routes: [{ expert: 'e1', slot: 0 }] }];
  input.capacities = { e1: 1, e2: 1 };
  input.workers = {
    w1: { status: 'dead', experts: ['e1'], deadSince: 800 }, // 200ms < 500ms window
    w2: { status: 'alive', experts: ['e2'] },
  };
  input.expertToWorker = { e1: 'w1', e2: 'w2' };
  input.received = [
    { tokenId: 't1', expert: 'e1', slot: 0, status: 'queued', gradientVersion: 5 },
  ];

  const plan = planRecovery(input);
  assert.equal(plan.reroutes.length, 0);
  assert.equal(plan.waits.length, 1);
  assert.equal(plan.waits[0].waitMs, 300);
  assert.equal(plan.decisions[0].action, Action.WAIT);
});

test('capacity overflow: tokens beyond spare capacity become unrecoverable', () => {
  const input = baseInput();
  input.tokens = [
    { id: 't1', routes: [{ expert: 'e1', slot: 0 }] },
    { id: 't2', routes: [{ expert: 'e1', slot: 0 }] },
    { id: 't3', routes: [{ expert: 'e1', slot: 0 }] },
  ];
  input.capacities = { e1: 3, e2: 2 }; // e2 can absorb only 2 of 3
  input.workers = {
    w1: { status: 'dead', experts: ['e1'], deadSince: 0 },
    w2: { status: 'alive', experts: ['e2'] },
  };
  input.expertToWorker = { e1: 'w1', e2: 'w2' };
  input.received = [
    { tokenId: 't1', expert: 'e1', slot: 0, status: 'queued', gradientVersion: 5 },
    { tokenId: 't2', expert: 'e1', slot: 0, status: 'queued', gradientVersion: 5 },
    { tokenId: 't3', expert: 'e1', slot: 0, status: 'queued', gradientVersion: 5 },
  ];

  const plan = planRecovery(input);
  assert.equal(plan.reroutes.length, 2);
  assert.equal(plan.unrecoverable.length, 1);
  assert.equal(plan.unrecoverable[0].tokenId, 't3');
  assert.equal(plan.unrecoverable[0].reason, 'no-alive-expert-with-capacity');
  assert.ok(plan.plannedLoad.e2 <= 2);
});

test('partial completion: done tokens are kept, stale versions roll back', () => {
  const input = baseInput();
  input.tokens = [
    { id: 't1', routes: [{ expert: 'e1', slot: 0 }] },
    { id: 't2', routes: [{ expert: 'e1', slot: 0 }] },
    { id: 't3', routes: [{ expert: 'e2', slot: 0 }] },
  ];
  input.capacities = { e1: 3, e2: 3 };
  input.workers = {
    w1: { status: 'dead', experts: ['e1'], deadSince: 0 },
    w2: { status: 'alive', experts: ['e2'] },
  };
  input.expertToWorker = { e1: 'w1', e2: 'w2' };
  input.received = [
    // Computed before the crash, but on an old gradient version.
    { tokenId: 't1', expert: 'e1', slot: 0, status: 'done', gradientVersion: 4 },
    // Never finished.
    { tokenId: 't2', expert: 'e1', slot: 0, status: 'computing', gradientVersion: 5 },
    // Healthy expert, current version: untouched.
    { tokenId: 't3', expert: 'e2', slot: 0, status: 'done', gradientVersion: 5 },
  ];

  const plan = planRecovery(input);
  // Stale done work must be recomputed -> reroute, and its step rolled back.
  assert.deepEqual(plan.rollbackSteps, [4]);
  assert.equal(plan.reroutes.length, 2);
  assert.deepEqual(
    plan.reroutes.map((r) => r.tokenId).sort(),
    ['t1', 't2'],
  );
  // t3 was already done at the current version: no decision, load counted.
  assert.ok(!plan.decisions.some((d) => d.tokenId === 't3'));
  assert.equal(plan.plannedLoad.e2, 1 + 2); // t3 done + t1,t2 rerouted in
});

test('duplicate routes: token computed twice for one slot is rolled back', () => {
  const input = baseInput();
  input.tokens = [
    { id: 't1', routes: [{ expert: 'e1', slot: 0 }, { expert: 'e2', slot: 1 }] },
  ];
  input.capacities = { e1: 1, e2: 1, e3: 1 };
  input.workers = {
    w1: { status: 'alive', experts: ['e1', 'e2'] },
    w2: { status: 'alive', experts: ['e3'] },
  };
  input.expertToWorker = { e1: 'w1', e2: 'w1', e3: 'w2' };
  input.received = [
    // Router bug: same token+slot delivered to (and computed by) two experts.
    { tokenId: 't1', expert: 'e1', slot: 0, status: 'done', gradientVersion: 5 },
    { tokenId: 't1', expert: 'e3', slot: 0, status: 'done', gradientVersion: 5 },
  ];

  const plan = planRecovery(input);
  assert.equal(plan.reroutes.length, 0);
  assert.deepEqual(plan.rollbackSteps, [5]);
  const dup = plan.decisions.find((d) => d.reason === 'duplicate-route');
  assert.ok(dup, 'expected a duplicate-route decision');
  assert.equal(dup.action, Action.FAIL);
  assert.equal(dup.expert, 'e3');
});

test('duplicate slot in route table is rejected, never double-computed', () => {
  const input = baseInput();
  input.tokens = [
    {
      id: 't1',
      routes: [
        { expert: 'e1', slot: 0 },
        { expert: 'e2', slot: 0 }, // same slot twice
      ],
    },
  ];
  input.capacities = { e1: 1, e2: 1 };
  input.workers = { w1: { status: 'alive', experts: ['e1', 'e2'] } };
  input.expertToWorker = { e1: 'w1', e2: 'w1' };

  const plan = planRecovery(input);
  const dup = plan.decisions.find((d) => d.reason === 'duplicate-route-slot');
  assert.ok(dup);
  assert.equal(dup.action, Action.FAIL);
  assert.deepEqual(plan.rollbackSteps, [5]);
});

test('reroute never targets a dead expert or an expert already holding the token', () => {
  const input = baseInput();
  input.tokens = [
    { id: 't1', routes: [{ expert: 'e1', slot: 0 }, { expert: 'e2', slot: 1 }] },
  ];
  input.capacities = { e1: 1, e2: 1, e3: 1 };
  input.workers = {
    w1: { status: 'dead', experts: ['e1'], deadSince: 0 },
    w2: { status: 'alive', experts: ['e2', 'e3'] },
  };
  input.expertToWorker = { e1: 'w1', e2: 'w2', e3: 'w2' };
  input.received = [
    { tokenId: 't1', expert: 'e1', slot: 0, status: 'queued', gradientVersion: 5 },
    // t1 already computed its slot-1 copy on e2.
    { tokenId: 't1', expert: 'e2', slot: 1, status: 'done', gradientVersion: 5 },
  ];

  const plan = planRecovery(input);
  assert.equal(plan.reroutes.length, 1);
  // e2 is full (holds the slot-1 copy) so the reroute must land on e3.
  assert.equal(plan.reroutes[0].toExpert, 'e3');
  assert.notEqual(plan.reroutes[0].toExpert, 'e1');
});
