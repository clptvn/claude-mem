import { describe, expect, test } from 'bun:test';
import { Database } from 'bun:sqlite';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseLongMemEval } from '../../benchmarks/memory-retrieval/datasets.js';
import { calculateRetrievalMetrics } from '../../benchmarks/memory-retrieval/metrics.js';
import { scoreMemoryAgentBench } from '../../benchmarks/memory-retrieval/memoryagentbench.js';

describe('memory benchmark adapters and metrics', () => {
  test('parses LongMemEval session and turn evidence without cross-query leakage', () => {
    const entry = {
      question_id: 'q1',
      question_type: 'temporal-reasoning',
      question: 'Which port was selected?',
      answer: '37701',
      question_date: '2026-07-29',
      haystack_session_ids: ['answer_session', 'distractor_session'],
      haystack_dates: ['2026-07-28', '2026-07-27'],
      answer_session_ids: ['answer_session'],
      haystack_sessions: [
        [
          { role: 'user', content: 'The port is 37701.', has_answer: true },
          { role: 'assistant', content: 'Acknowledged.' },
        ],
        [
          { role: 'user', content: 'The navbar is blue.', has_answer: false },
          { role: 'assistant', content: 'Acknowledged.' },
        ],
      ],
    };
    const session = parseLongMemEval([entry], { granularity: 'session' });
    const turn = parseLongMemEval([entry], { granularity: 'turn' });

    expect(session.queries[0].expected).toEqual(['q1:answer_session']);
    expect(session.queries[0].candidateIds).toHaveLength(2);
    expect(session.documents[0].text).not.toContain('Acknowledged');
    expect(turn.queries[0].expected).toEqual(['q1:answer_session_1']);
    expect(turn.queries[0].candidateIds).toHaveLength(2);
  });

  test('treats an answer-named LongMemEval session with no labeled answer turn as abstention', () => {
    const dataset = parseLongMemEval([{
      question_id: 'q-abstain',
      question_type: 'abstention',
      question: 'What was never discussed?',
      answer: 'I do not know.',
      question_date: '2026-07-29',
      haystack_session_ids: ['answer_session'],
      haystack_dates: ['2026-07-28'],
      answer_session_ids: [],
      haystack_sessions: [[
        { role: 'user', content: 'Unrelated context.', has_answer: false },
      ]],
    }], { granularity: 'session' });

    expect(dataset.queries[0].expected).toEqual([]);
  });

  test('computes all-evidence recall, nDCG, abstention, and forbidden selection', () => {
    const queries = [
      {
        id: 'answerable',
        category: 'multi-hop',
        text: 'query',
        expected: ['a', 'b'],
        forbidden: ['stale'],
      },
      {
        id: 'unknown',
        category: 'abstention',
        text: 'unknown',
        expected: [],
      },
    ];
    const selected = new Map([
      ['answerable', [
        {
          id: 'a',
          session: 's1',
          createdAtEpoch: 1,
          text: 'a',
          similarity: 0.9,
        },
        {
          id: 'b',
          session: 's2',
          createdAtEpoch: 2,
          text: 'b',
          similarity: 0.8,
        },
      ]],
      ['unknown', []],
    ]);
    const metrics = calculateRetrievalMetrics(
      queries,
      selected,
      new Map([['answerable', 1], ['unknown', 2]]),
      [1, 5],
    );
    expect(metrics.recallAnyAtK['1']).toBe(1);
    expect(metrics.recallAllAtK['1']).toBe(0);
    expect(metrics.recallAllAtK['5']).toBe(1);
    expect(metrics.ndcgAtK['5']).toBe(1);
    expect(metrics.abstentionAccuracy).toBe(1);
    expect(metrics.forbiddenSelectionRate).toBe(0);
  });

  test('scores deterministic MemoryAgentBench-compatible predictions', () => {
    const score = scoreMemoryAgentBench([
      {
        task_id: 'eventqa:0',
        competency: 'Accurate_Retrieval',
        source: 'eventqa_full',
        context: 'context',
        questions: ['What happened?'],
        answers: [['alpha', 'beta']],
        metadata: {},
      },
    ], [
      {
        task_id: 'eventqa:0',
        question_index: 0,
        prediction: 'The answer contains alpha and beta.',
      },
    ]);
    expect(score.scored).toBe(1);
    expect(score.substringMatch).toBe(1);
    expect(score.eventQaAllFacts).toBe(1);
  });

  test('keeps the benchmark cache schema in a checked-in migration', () => {
    const database = new Database(':memory:');
    try {
      database.exec(readFileSync(join(
        process.cwd(),
        'benchmarks/memory-retrieval/migrations/001_create_embedding_cache.sql',
      ), 'utf8'));
      const tables = database.query<{ name: string }, []>(
        "SELECT name FROM sqlite_master WHERE type = 'table'",
      ).all();
      expect(tables.map(table => table.name)).toContain('embedding_cache');
    } finally {
      database.close();
    }
  });
});
