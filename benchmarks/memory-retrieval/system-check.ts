import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

interface SystemCheckResult {
  schemaVersion: 1;
  createdAt: string;
  serviceUrl: string;
  workerUrl: string;
  collection: string;
  checks: Record<string, { passed: boolean; detail: unknown }>;
  concurrency: {
    requests: number;
    totalMs: number;
    requestsPerSecond: number;
  };
  cleanup: { attempted: boolean; passed: boolean };
}

async function jsonFetch(
  url: string,
  init?: RequestInit,
): Promise<{ status: number; body: unknown }> {
  const response = await fetch(url, {
    ...init,
    signal: AbortSignal.timeout(180_000),
  });
  const text = await response.text();
  let body: unknown = text;
  try {
    body = JSON.parse(text);
  } catch {
    // Keep text error bodies intact.
  }
  if (!response.ok) {
    throw new Error(`${url} failed (${response.status}): ${text}`);
  }
  return { status: response.status, body };
}

export async function runSystemCheck(options: {
  serviceUrl: string;
  workerUrl: string;
  runsDir: string;
}): Promise<{ result: SystemCheckResult; artifactPath: string }> {
  const suffix = `${process.pid}-${Date.now()}`;
  const collection = `cm__benchmark_${suffix}`;
  const checks: SystemCheckResult['checks'] = {};
  let cleanupAttempted = false;
  let cleanupPassed = false;
  const call = async (toolName: string, args: Record<string, unknown>) => {
    const response = await jsonFetch(`${options.serviceUrl}/v1/vector/call`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ tool_name: toolName, arguments: args }),
    });
    return (response.body as { result: unknown }).result;
  };

  const health = (await jsonFetch(`${options.serviceUrl}/healthz`)).body as {
    model?: { state?: string };
    privacy?: { chroma_anonymized_telemetry?: boolean };
    durable_write_queue?: { pending?: number; processing?: number; failed?: number };
  };
  checks.modelReady = {
    passed: health.model?.state === 'ready',
    detail: health.model ?? null,
  };
  checks.telemetryDisabled = {
    passed: health.privacy?.chroma_anonymized_telemetry === false,
    detail: health.privacy ?? null,
  };
  checks.queueHealthy = {
    passed:
      (health.durable_write_queue?.failed ?? 0) === 0
      && (health.durable_write_queue?.pending ?? 0) === 0
      && (health.durable_write_queue?.processing ?? 0) === 0,
    detail: health.durable_write_queue ?? null,
  };

  try {
    const worker = (await jsonFetch(`${options.workerUrl}/api/health`)).body as {
      status?: string;
      initialized?: boolean;
      mcpReady?: boolean;
      ai?: { provider?: string };
    };
    checks.workerReady = {
      passed: worker.status === 'ok' && worker.initialized === true && worker.mcpReady === true,
      detail: {
        status: worker.status,
        initialized: worker.initialized,
        mcpReady: worker.mcpReady,
        provider: worker.ai?.provider,
      },
    };
  } catch (error) {
    checks.workerReady = {
      passed: false,
      detail: error instanceof Error ? error.message : String(error),
    };
  }

  try {
    await call('chroma_create_collection', { collection_name: collection });
    await call('chroma_add_documents', {
      collection_name: collection,
      ids: ['shared-memory', 'unrelated'],
      documents: [
        'Claude and Codex share one durable local memory service backed by Nemotron embeddings.',
        'A sourdough starter is fed with flour and water.',
      ],
      metadatas: [
        { benchmark: true, doc_type: 'observation' },
        { benchmark: true, doc_type: 'observation' },
      ],
    });
    const count = await call('chroma_get_collection_count', {
      collection_name: collection,
    });
    checks.durableWrite = { passed: count === 2, detail: { count } };

    const query = await call('chroma_query_documents', {
      collection_name: collection,
      query_texts: ['How do the coding agents remember together?'],
      n_results: 2,
      include: ['documents', 'metadatas', 'distances'],
    }) as { ids?: string[][]; distances?: number[][] };
    checks.semanticRanking = {
      passed: query.ids?.[0]?.[0] === 'shared-memory',
      detail: {
        ids: query.ids?.[0] ?? [],
        distances: query.distances?.[0] ?? [],
      },
    };
  } finally {
    cleanupAttempted = true;
    try {
      await call('chroma_delete_collection', { collection_name: collection });
      const collections = await call('chroma_list_collections', {}) as string[];
      cleanupPassed = !collections.includes(collection);
    } catch {
      cleanupPassed = false;
    }
  }

  const concurrencyRequests = 12;
  const concurrencyStarted = performance.now();
  await Promise.all(Array.from({ length: concurrencyRequests }, (_, index) =>
    jsonFetch(`${options.serviceUrl}/v1/embeddings`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        input: `Concurrent benchmark request ${index}: shared memory embedding queue`,
        input_type: 'query',
      }),
    })
  ));
  const concurrencyMs = performance.now() - concurrencyStarted;

  const result: SystemCheckResult = {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    serviceUrl: options.serviceUrl,
    workerUrl: options.workerUrl,
    collection,
    checks,
    concurrency: {
      requests: concurrencyRequests,
      totalMs: Number(concurrencyMs.toFixed(3)),
      requestsPerSecond: Number((concurrencyRequests / (concurrencyMs / 1_000)).toFixed(3)),
    },
    cleanup: { attempted: cleanupAttempted, passed: cleanupPassed },
  };
  mkdirSync(options.runsDir, { recursive: true });
  const artifactPath = join(
    options.runsDir,
    `${new Date().toISOString().replace(/[:.]/g, '-')}-system-check.json`,
  );
  writeFileSync(artifactPath, `${JSON.stringify(result, null, 2)}\n`);

  console.table(Object.entries(checks).map(([check, value]) => ({
    check,
    passed: value.passed,
  })));
  console.log(
    `Concurrent embeddings: ${result.concurrency.requestsPerSecond.toFixed(2)} req/s ` +
    `(${concurrencyRequests} requests)`,
  );
  console.log(`Ephemeral collection cleanup: ${cleanupPassed ? 'passed' : 'FAILED'} (${collection})`);
  console.log(`Artifact: ${artifactPath}`);

  if (Object.values(checks).some(check => !check.passed) || !cleanupPassed) {
    throw new Error('One or more local memory system checks failed');
  }
  return { result, artifactPath };
}
