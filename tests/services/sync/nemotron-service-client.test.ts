import { afterEach, beforeEach, describe, expect, it, mock } from 'bun:test';
import { NemotronServiceClient } from '../../../src/services/sync/NemotronServiceClient.js';
import { ChromaUnavailableError } from '../../../src/services/worker/search/errors.js';

const originalFetch = globalThis.fetch;
const originalEnv = {
  provider: process.env.CLAUDE_MEM_EMBEDDING_PROVIDER,
  url: process.env.CLAUDE_MEM_NEMOTRON_URL,
  autoStart: process.env.CLAUDE_MEM_NEMOTRON_AUTO_START,
  timeout: process.env.CLAUDE_MEM_NEMOTRON_REQUEST_TIMEOUT_MS,
};

describe('NemotronServiceClient', () => {
  beforeEach(() => {
    process.env.CLAUDE_MEM_EMBEDDING_PROVIDER = 'nemotron';
    process.env.CLAUDE_MEM_NEMOTRON_URL = 'http://127.0.0.1:37901';
    process.env.CLAUDE_MEM_NEMOTRON_AUTO_START = 'false';
    process.env.CLAUDE_MEM_NEMOTRON_REQUEST_TIMEOUT_MS = '1000';
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    for (const [key, value] of Object.entries({
      CLAUDE_MEM_EMBEDDING_PROVIDER: originalEnv.provider,
      CLAUDE_MEM_NEMOTRON_URL: originalEnv.url,
      CLAUDE_MEM_NEMOTRON_AUTO_START: originalEnv.autoStart,
      CLAUDE_MEM_NEMOTRON_REQUEST_TIMEOUT_MS: originalEnv.timeout,
    })) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  it('preserves the chroma-mcp call contract over HTTP', async () => {
    const fetchMock = mock(async (_url: string | URL | Request, init?: RequestInit) => {
      const payload = JSON.parse(String(init?.body));
      expect(payload).toEqual({
        tool_name: 'chroma_list_collections',
        arguments: { limit: 1 },
      });
      return new Response(JSON.stringify({ result: ['cm__project'] }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });
    globalThis.fetch = fetchMock as typeof fetch;

    const result = await NemotronServiceClient.getInstance().callTool(
      'chroma_list_collections',
      { limit: 1 },
    );

    expect(result).toEqual(['cm__project']);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('wraps service failures as the existing semantic-search availability error', async () => {
    globalThis.fetch = mock(async () => new Response(
      JSON.stringify({ detail: 'model failed to load' }),
      { status: 503, headers: { 'content-type': 'application/json' } },
    )) as typeof fetch;

    await expect(
      NemotronServiceClient.getInstance().callTool('chroma_query_documents', {
        collection_name: 'cm__project',
        query_texts: ['memory'],
      }),
    ).rejects.toBeInstanceOf(ChromaUnavailableError);
  });

  it('uses model readiness rather than process liveness for health checks', async () => {
    const fetchMock = mock(async (url: string | URL | Request) => {
      expect(String(url)).toBe('http://127.0.0.1:37901/readyz');
      return new Response(JSON.stringify({ ready: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });
    globalThis.fetch = fetchMock as typeof fetch;

    expect(await NemotronServiceClient.getInstance().isHealthy()).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
