import { describe, expect, test } from 'bun:test';
import {
  applyRetrievalPolicy,
  calculateEffectiveMinimumSimilarity,
} from '../../src/services/worker/context-retrieval-policy.js';

function candidate(overrides: Partial<{
  memorySessionId: string;
  title: string;
  body: string;
  stateKey: string;
  createdAtEpoch: number;
  similarity: number;
}> = {}) {
  return {
    memorySessionId: 'session-a',
    title: 'worker port',
    body: 'The worker port is configured.',
    createdAtEpoch: 100,
    similarity: 0.8,
    ...overrides,
  };
}

describe('shared automatic retrieval policy', () => {
  test('uses the stricter of the absolute floor and relative confidence window', () => {
    expect(calculateEffectiveMinimumSimilarity([
      { similarity: 0.80 },
      { similarity: 0.55 },
    ], {
      minimumSimilarity: 0.18,
      relativeBand: 0.24,
    })).toBeCloseTo(0.56);
  });

  test('selects current or previous state from the same stable key', () => {
    const old = candidate({
      body: 'The worker port was 37700.',
      stateKey: 'worker.port',
      createdAtEpoch: 100,
      similarity: 0.81,
    });
    const current = candidate({
      memorySessionId: 'session-b',
      body: 'The worker port is 37701.',
      stateKey: 'worker.port',
      createdAtEpoch: 200,
      similarity: 0.80,
    });

    const now = applyRetrievalPolicy([old, current], {
      query: 'What port does the worker use now?',
      minimumSimilarity: 0,
      relativeBand: 2,
    });
    const previous = applyRetrievalPolicy([old, current], {
      query: 'What port did the worker previously use?',
      minimumSimilarity: 0,
      relativeBand: 2,
    });

    expect(now.candidates).toEqual([current]);
    expect(previous.candidates).toEqual([old]);
  });

  test('caps one session and suppresses near-duplicates', () => {
    const first = candidate({ title: 'queue result', body: 'The queue drained successfully.' });
    const duplicate = candidate({
      title: 'queue result duplicate',
      body: 'The queue drained successfully.',
      similarity: 0.79,
    });
    const third = candidate({
      title: 'queue follow-up',
      body: 'A separate follow-up from the same session.',
      similarity: 0.78,
    });
    const selected = applyRetrievalPolicy([first, duplicate, third], {
      query: 'What happened to the queue?',
      minimumSimilarity: 0,
      relativeBand: 2,
      maxPerSession: 1,
    });
    expect(selected.candidates).toHaveLength(1);
    expect(selected.rejectedRedundant).toBe(2);
  });
});
