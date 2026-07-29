import { describe, it, expect, mock } from 'bun:test';
import { SearchManager } from '../../src/services/worker/SearchManager.js';

describe('SearchManager platform-scoped Chroma hydration', () => {
  it('normalizes date_from/date_to filters into dateRange for worker search', async () => {
    const searchObservations = mock(() => []);
    const manager = new SearchManager(
      {
        searchObservations,
        searchSessions: mock(() => []),
        searchUserPrompts: mock(() => []),
      } as any,
      {} as any,
      null,
      {} as any,
      {} as any,
    );

    await manager.search({
      type: 'observations',
      date_from: '2025-01-01',
      date_to: '2025-01-31',
      format: 'json',
    });

    expect(searchObservations).toHaveBeenCalledWith(undefined, expect.objectContaining({
      dateRange: {
        start: '2025-01-01',
        end: '2025-01-31',
      },
    }));
  });

  it('passes platformSource into Chroma observation where filter and SQLite hydration', async () => {
    const observation = {
      id: 5,
      memory_session_id: 'cursor-memory-id',
      project: 'search-project',
      text: null,
      type: 'discovery',
      title: 'cursor overlap observation',
      subtitle: null,
      facts: '[]',
      narrative: 'cursor overlap narrative',
      concepts: '[]',
      files_read: '[]',
      files_modified: '[]',
      prompt_number: 1,
      discovery_tokens: 0,
      created_at: new Date().toISOString(),
      created_at_epoch: Date.now(),
    };
    const getObservationsByIds = mock(() => [observation]);
    const queryChroma = mock(() => Promise.resolve({
      ids: [observation.id],
      distances: [0.1],
      metadatas: [{
        sqlite_id: observation.id,
        doc_type: 'observation',
        project: 'search-project',
        platform_source: 'cursor',
        created_at_epoch: Date.now(),
      }],
    }));

    const manager = new SearchManager(
      {
        searchObservations: mock(() => []),
        searchSessions: mock(() => []),
        searchUserPrompts: mock(() => []),
      } as any,
      {
        getObservationsByIds,
        getSessionSummariesByIds: mock(() => []),
        getUserPromptsByIds: mock(() => []),
      } as any,
      { queryChroma } as any,
      {} as any,
      {} as any,
    );

    const result = await manager.search({
      query: 'overlap',
      type: 'observations',
      project: 'search-project',
      platformSource: 'cursor',
      format: 'json',
      limit: 10,
    });

    expect(queryChroma).toHaveBeenCalledWith('overlap', 100, {
      $and: [
        { doc_type: 'observation' },
        { $or: [{ project: 'search-project' }, { merged_into_project: 'search-project' }] },
        { platform_source: 'cursor' },
      ],
    });
    expect(getObservationsByIds).toHaveBeenCalledWith([observation.id], expect.objectContaining({
      platformSource: 'cursor',
      project: 'search-project',
    }));
    expect(result.observations).toEqual([observation]);
  });

  it('passes platformSource into Chroma session where filter and SQLite hydration', async () => {
    const session = {
      id: 6,
      memory_session_id: 'cursor-memory-id',
      project: 'search-project',
      request: 'cursor overlap session',
      investigated: null,
      learned: null,
      completed: null,
      next_steps: null,
      files_read: null,
      files_edited: null,
      notes: null,
      prompt_number: 1,
      discovery_tokens: 0,
      created_at: new Date().toISOString(),
      created_at_epoch: Date.now(),
    };
    const getSessionSummariesByIds = mock(() => [session]);
    const queryChroma = mock(() => Promise.resolve({
      ids: [session.id],
      distances: [0.1],
      metadatas: [{
        sqlite_id: session.id,
        doc_type: 'session_summary',
        project: 'search-project',
        platform_source: 'cursor',
        created_at_epoch: Date.now(),
      }],
    }));

    const manager = new SearchManager(
      {
        searchObservations: mock(() => []),
        searchSessions: mock(() => []),
        searchUserPrompts: mock(() => []),
      } as any,
      {
        getObservationsByIds: mock(() => []),
        getSessionSummariesByIds,
        getUserPromptsByIds: mock(() => []),
      } as any,
      { queryChroma } as any,
      {} as any,
      {} as any,
    );

    const result = await manager.search({
      query: 'overlap',
      type: 'sessions',
      project: 'search-project',
      platformSource: 'cursor',
      format: 'json',
      limit: 10,
    });

    expect(queryChroma).toHaveBeenCalledWith('overlap', 100, {
      $and: [
        { doc_type: 'session_summary' },
        { $or: [{ project: 'search-project' }, { merged_into_project: 'search-project' }] },
        { platform_source: 'cursor' },
      ],
    });
    expect(getSessionSummariesByIds).toHaveBeenCalledWith([session.id], {
      orderBy: 'date_desc',
      limit: 10,
      project: 'search-project',
      platformSource: 'cursor',
    });
    expect(result.sessions).toEqual([session]);
  });

  it('passes platformSource into Chroma prompt SQLite hydration', async () => {
    const prompt = {
      id: 7,
      content_session_id: 'shared-raw-id',
      prompt_number: 1,
      prompt_text: 'cursor overlap prompt',
      project: 'search-project',
      platform_source: 'cursor',
      created_at: new Date().toISOString(),
      created_at_epoch: Date.now(),
    };
    const getUserPromptsByIds = mock(() => [prompt]);
    const queryChroma = mock(() => Promise.resolve({
      ids: [prompt.id],
      distances: [0.1],
      metadatas: [{
        sqlite_id: prompt.id,
        doc_type: 'user_prompt',
        project: 'search-project',
        platform_source: 'cursor',
        created_at_epoch: Date.now(),
      }],
    }));

    const manager = new SearchManager(
      {
        searchObservations: mock(() => []),
        searchSessions: mock(() => []),
        searchUserPrompts: mock(() => []),
      } as any,
      {
        getObservationsByIds: mock(() => []),
        getSessionSummariesByIds: mock(() => []),
        getUserPromptsByIds,
      } as any,
      { queryChroma } as any,
      {} as any,
      {} as any,
    );

    const result = await manager.search({
      query: 'overlap',
      type: 'prompts',
      project: 'search-project',
      platformSource: 'cursor',
      format: 'json',
      limit: 10,
    });

    expect(getUserPromptsByIds).toHaveBeenCalledWith([prompt.id], {
      orderBy: 'date_desc',
      limit: 10,
      project: 'search-project',
      platformSource: 'cursor',
    });
    expect(result.prompts).toEqual([prompt]);
  });

  it('passes platformSource into getTimelineByQuery auto-mode hydration', async () => {
    const observation = {
      id: 8,
      memory_session_id: 'cursor-memory-id',
      project: 'search-project',
      text: null,
      type: 'discovery',
      title: 'cursor timeline anchor',
      subtitle: null,
      facts: '[]',
      narrative: 'cursor timeline narrative',
      concepts: '[]',
      files_read: '[]',
      files_modified: '[]',
      prompt_number: 1,
      discovery_tokens: 0,
      created_at: new Date().toISOString(),
      created_at_epoch: Date.now(),
    };
    const searchObservations = mock(() => [observation]);
    const getTimelineAroundObservation = mock(() => ({
      observations: [],
      sessions: [],
      prompts: [],
    }));

    const manager = new SearchManager(
      {
        searchObservations,
        searchSessions: mock(() => []),
        searchUserPrompts: mock(() => []),
      } as any,
      {
        getObservationsByIds: mock(() => []),
        getSessionSummariesByIds: mock(() => []),
        getUserPromptsByIds: mock(() => []),
        getTimelineAroundObservation,
      } as any,
      null,
      {} as any,
      { filterByDepth: mock(() => []) } as any,
    );

    await manager.getTimelineByQuery({
      query: 'timeline',
      mode: 'auto',
      project: 'search-project',
      platform_source: 'cursor',
    });

    expect(searchObservations).toHaveBeenCalledWith('timeline', {
      project: 'search-project',
      platformSource: 'cursor',
      limit: 1,
    });
    expect(getTimelineAroundObservation).toHaveBeenCalledWith(
      observation.id,
      observation.created_at_epoch,
      10,
      10,
      'search-project',
      'cursor',
    );
  });

  it('falls back to scoped SQLite/FTS when platform-scoped Chroma returns zero matches', async () => {
    const observation = {
      id: 9,
      memory_session_id: 'cursor-memory-id',
      project: 'search-project',
      text: null,
      type: 'discovery',
      title: 'cursor fallback observation',
      subtitle: null,
      facts: '[]',
      narrative: 'cursor fallback narrative',
      concepts: '[]',
      files_read: '[]',
      files_modified: '[]',
      prompt_number: 1,
      discovery_tokens: 0,
      created_at: new Date().toISOString(),
      created_at_epoch: Date.now(),
    };
    const session = {
      id: 10,
      memory_session_id: 'cursor-memory-id',
      project: 'search-project',
      request: 'cursor fallback session',
      investigated: null,
      learned: null,
      completed: null,
      next_steps: null,
      files_read: null,
      files_edited: null,
      notes: null,
      prompt_number: 1,
      discovery_tokens: 0,
      created_at: new Date().toISOString(),
      created_at_epoch: Date.now(),
    };
    const prompt = {
      id: 11,
      content_session_id: 'shared-raw-id',
      prompt_number: 1,
      prompt_text: 'cursor fallback prompt',
      project: 'search-project',
      platform_source: 'cursor',
      created_at: new Date().toISOString(),
      created_at_epoch: Date.now(),
    };
    const searchObservations = mock(() => [observation]);
    const searchSessions = mock(() => [session]);
    const searchUserPrompts = mock(() => [prompt]);
    const queryChroma = mock(() => Promise.resolve({
      ids: [],
      distances: [],
      metadatas: [],
    }));

    const manager = new SearchManager(
      {
        searchObservations,
        searchSessions,
        searchUserPrompts,
      } as any,
      {
        getObservationsByIds: mock(() => []),
        getSessionSummariesByIds: mock(() => []),
        getUserPromptsByIds: mock(() => []),
      } as any,
      { queryChroma } as any,
      {} as any,
      {} as any,
    );
    const telemetry = {};

    const result = await manager.search({
      query: 'legacy metadata',
      project: 'search-project',
      platformSource: 'cursor',
      format: 'json',
      limit: 10,
    }, telemetry);

    expect(searchObservations).toHaveBeenCalledWith('legacy metadata', expect.objectContaining({
      project: 'search-project',
      platformSource: 'cursor',
    }));
    expect(searchSessions).toHaveBeenCalledWith('legacy metadata', expect.objectContaining({
      project: 'search-project',
      platformSource: 'cursor',
    }));
    expect(searchUserPrompts).toHaveBeenCalledWith('legacy metadata', expect.objectContaining({
      project: 'search-project',
      platformSource: 'cursor',
    }));
    expect(result).toEqual(expect.objectContaining({
      observations: [observation],
      sessions: [session],
      prompts: [prompt],
      totalResults: 3,
    }));
    expect(telemetry).toEqual(expect.objectContaining({
      result_count: 3,
      search_strategy: 'fts',
      chroma_available: true,
      fallback_reason: 'chroma_error',
    }));
  });

  it('keeps unscoped Chroma zero matches final without SQLite/FTS fallback', async () => {
    const searchObservations = mock(() => []);
    const searchSessions = mock(() => []);
    const searchUserPrompts = mock(() => []);
    const queryChroma = mock(() => Promise.resolve({
      ids: [],
      distances: [],
      metadatas: [],
    }));

    const manager = new SearchManager(
      {
        searchObservations,
        searchSessions,
        searchUserPrompts,
      } as any,
      {
        getObservationsByIds: mock(() => []),
        getSessionSummariesByIds: mock(() => []),
        getUserPromptsByIds: mock(() => []),
      } as any,
      { queryChroma } as any,
      {} as any,
      {} as any,
    );
    const telemetry = {};

    const result = await manager.search({
      query: 'legacy metadata',
      format: 'json',
    }, telemetry);

    expect(searchObservations).not.toHaveBeenCalled();
    expect(searchSessions).not.toHaveBeenCalled();
    expect(searchUserPrompts).not.toHaveBeenCalled();
    expect(result).toEqual(expect.objectContaining({
      observations: [],
      sessions: [],
      prompts: [],
      totalResults: 0,
    }));
    expect(telemetry).toEqual(expect.objectContaining({
      result_count: 0,
      search_strategy: 'chroma',
      chroma_available: true,
      fallback_reason: 'none',
    }));
  });
});

describe('SearchManager automatic context retrieval', () => {
  const createdAt = (epoch: number) => new Date(epoch).toISOString();

  it('abstains when every nearest neighbor is below the configured similarity floor', async () => {
    const queryChroma = mock(() => Promise.resolve({
      ids: [1, 2],
      distances: [0.86, 0.91],
      metadatas: [
        { sqlite_id: 1, doc_type: 'observation', field_type: 'fact' },
        { sqlite_id: 2, doc_type: 'session_summary', field_type: 'learned' },
      ],
    }));
    const manager = new SearchManager(
      {} as any,
      {
        getObservationsByIds: mock(() => {
          throw new Error('weak candidates must not be hydrated');
        }),
        getSessionSummariesByIds: mock(() => {
          throw new Error('weak candidates must not be hydrated');
        }),
      } as any,
      { queryChroma } as any,
      {} as any,
      {} as any,
    );

    const result = await manager.retrieveContext({
      query: 'compose a haiku about purple otters on Mars',
      project: 'memory-project',
      minimumSimilarity: 0.18,
    });

    expect(result.candidates).toEqual([]);
    expect(result.considered).toBe(2);
    expect(result.rejectedLowConfidence).toBe(2);
  });

  it('globally ranks observations and summaries while preserving the matched vector field', async () => {
    const now = Date.now();
    const queryChroma = mock(() => Promise.resolve({
      ids: [20, 10],
      distances: [0.22, 0.28],
      metadatas: [
        {
          sqlite_id: 20,
          doc_type: 'session_summary',
          field_type: 'learned',
          created_at_epoch: now,
        },
        {
          sqlite_id: 10,
          doc_type: 'observation',
          field_type: 'fact',
          created_at_epoch: now - 1000,
        },
      ],
    }));
    const manager = new SearchManager(
      {} as any,
      {
        getObservationsByIds: mock(() => [{
          id: 10,
          memory_session_id: 'session-observation',
          title: 'Nemotron service decision',
          narrative: 'The embedding service stays local.',
          facts: JSON.stringify(['It listens only on loopback.']),
          created_at: createdAt(now - 1000),
          created_at_epoch: now - 1000,
        }]),
        getSessionSummariesByIds: mock(() => [{
          id: 20,
          memory_session_id: 'session-summary',
          request: 'Improve automatic memory retrieval',
          investigated: 'Weak-neighbor injection',
          learned: 'The system must abstain below a calibrated floor.',
          completed: null,
          next_steps: null,
          notes: null,
          created_at: createdAt(now),
          created_at_epoch: now,
        }]),
      } as any,
      { queryChroma } as any,
      {} as any,
      {} as any,
    );

    const result = await manager.retrieveContext({
      query: 'how should automatic memory retrieval abstain',
      project: 'memory-project',
      limit: 2,
    });

    expect(result.candidates.map(candidate => candidate.kind)).toEqual([
      'session_summary',
      'observation',
    ]);
    expect(result.candidates[0].matchedField).toBe('learned');
    expect(result.candidates[1].body).toContain('It listens only on loopback.');
  });

  it('keeps the newest repeated state unless the query explicitly asks for history', async () => {
    const oldEpoch = Date.now() - 86_400_000;
    const newEpoch = Date.now();
    const rows = [
      {
        id: 1,
        memory_session_id: 'old-session',
        title: 'Worker port configuration',
        narrative: 'The worker listens on port 37700.',
        facts: '[]',
        created_at: createdAt(oldEpoch),
        created_at_epoch: oldEpoch,
      },
      {
        id: 2,
        memory_session_id: 'new-session',
        title: 'Worker port configuration',
        narrative: 'The worker listens on port 37701.',
        facts: '[]',
        created_at: createdAt(newEpoch),
        created_at_epoch: newEpoch,
      },
    ];
    const queryChroma = mock(() => Promise.resolve({
      ids: [1, 2],
      distances: [0.20, 0.21],
      metadatas: [
        { sqlite_id: 1, doc_type: 'observation', field_type: 'narrative' },
        { sqlite_id: 2, doc_type: 'observation', field_type: 'narrative' },
      ],
    }));
    const manager = new SearchManager(
      {} as any,
      {
        getObservationsByIds: mock(() => rows),
        getSessionSummariesByIds: mock(() => []),
      } as any,
      { queryChroma } as any,
      {} as any,
      {} as any,
    );

    const current = await manager.retrieveContext({
      query: 'what port does the worker use',
      project: 'memory-project',
    });
    expect(current.candidates.map(candidate => candidate.id)).toEqual([2]);

    const historical = await manager.retrieveContext({
      query: 'what port did the worker previously use',
      project: 'memory-project',
    });
    expect(historical.candidates.map(candidate => candidate.id)).toEqual([1]);
  });
});
