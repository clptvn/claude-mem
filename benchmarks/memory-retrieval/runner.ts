import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  applyRetrievalPolicy,
  type RetrievalPolicyResult,
} from '../../src/services/worker/context-retrieval-policy.js';
import { datasetSha256 } from './datasets.js';
import { EmbeddingClient } from './embedding-client.js';
import { calculateRetrievalMetrics, rankByDotProduct } from './metrics.js';
import type {
  BenchmarkArtifact,
  BenchmarkDataset,
  BenchmarkQuery,
  PolicyRun,
  RankedDocument,
} from './types.js';

export interface BenchmarkRunOptions {
  serviceUrl: string;
  cachePath: string;
  runsDir: string;
  minimumSimilarity?: number;
  relativeBand?: number;
  limit?: number;
  maxPerSession?: number;
  ks?: number[];
  writeArtifact?: boolean;
}

interface Policy {
  name: string;
  description: string;
  select(ranked: RankedDocument[], query: BenchmarkQuery): {
    candidates: RankedDocument[];
    effectiveMinimumSimilarity: number | null;
  };
}

function policyCandidates(
  ranked: RankedDocument[],
): Array<RankedDocument & {
  memorySessionId: string;
  title: string;
  body: string;
}> {
  return ranked.map(document => ({
    ...document,
    memorySessionId: document.session,
    title: document.stateKey ?? document.id,
    body: document.text,
  }));
}

function productionPolicy(
  ranked: RankedDocument[],
  query: BenchmarkQuery,
  options: {
    minimumSimilarity: number;
    relativeBand: number;
    limit: number;
    maxPerSession: number;
    resolveFreshness: boolean;
    deduplicate: boolean;
    diversifySessions: boolean;
  },
): RetrievalPolicyResult<ReturnType<typeof policyCandidates>[number]> {
  return applyRetrievalPolicy(policyCandidates(ranked), {
    query: query.text,
    minimumSimilarity: options.minimumSimilarity,
    relativeBand: options.relativeBand,
    limit: options.limit,
    maxPerSession: options.maxPerSession,
    resolveFreshness: options.resolveFreshness,
    deduplicate: options.deduplicate,
    diversifySessions: options.diversifySessions,
  });
}

function buildPolicies(config: {
  minimumSimilarity: number;
  relativeBand: number;
  limit: number;
  maxPerSession: number;
  ks: number[];
}): Policy[] {
  const apply = (
    ranked: RankedDocument[],
    query: BenchmarkQuery,
    overrides: Partial<Parameters<typeof productionPolicy>[2]> = {},
  ) => {
    const result = productionPolicy(ranked, query, {
      ...config,
      resolveFreshness: true,
      deduplicate: true,
      diversifySessions: true,
      ...overrides,
    });
    return {
      candidates: result.candidates,
      effectiveMinimumSimilarity: result.effectiveMinimumSimilarity,
    };
  };
  return [
    {
      name: 'vector-ranking',
      description: 'Raw Nemotron ranking for official retrieval metrics; no injection policy.',
      select: ranked => ({
        candidates: ranked.slice(0, Math.max(...config.ks)),
        effectiveMinimumSimilarity: null,
      }),
    },
    {
      name: 'fixed-top-5',
      description: 'Legacy automatic injection baseline that never abstains.',
      select: ranked => ({
        candidates: ranked.slice(0, config.limit),
        effectiveMinimumSimilarity: null,
      }),
    },
    {
      name: 'threshold-only',
      description: 'Absolute cosine floor only; no relative window, freshness, dedupe, or diversity.',
      select: (ranked, query) => apply(ranked, query, {
        relativeBand: 2,
        resolveFreshness: false,
        deduplicate: false,
        diversifySessions: false,
      }),
    },
    {
      name: 'confidence-window',
      description: 'Absolute and relative cosine floors, without post-retrieval controls.',
      select: (ranked, query) => apply(ranked, query, {
        resolveFreshness: false,
        deduplicate: false,
        diversifySessions: false,
      }),
    },
    {
      name: 'calibrated-no-freshness',
      description: 'Production policy with temporal conflict resolution disabled.',
      select: (ranked, query) => apply(ranked, query, {
        resolveFreshness: false,
      }),
    },
    {
      name: 'calibrated-no-dedupe',
      description: 'Production policy with near-duplicate suppression disabled.',
      select: (ranked, query) => apply(ranked, query, {
        deduplicate: false,
      }),
    },
    {
      name: 'calibrated-production',
      description: 'Exact automatic retrieval policy used by SearchManager.',
      select: (ranked, query) => apply(ranked, query),
    },
  ];
}

function gitMetadata(): { gitCommit: string | null; gitDirty: boolean | null } {
  try {
    const commit = Bun.spawnSync(['git', 'rev-parse', 'HEAD'], {
      stdout: 'pipe',
      stderr: 'ignore',
    });
    const status = Bun.spawnSync(['git', 'status', '--porcelain'], {
      stdout: 'pipe',
      stderr: 'ignore',
    });
    return {
      gitCommit: commit.exitCode === 0
        ? commit.stdout.toString().trim()
        : null,
      gitDirty: status.exitCode === 0
        ? status.stdout.toString().trim().length > 0
        : null,
    };
  } catch {
    return { gitCommit: null, gitDirty: null };
  }
}

function runId(dataset: BenchmarkDataset): string {
  const suffix = createHash('sha256')
    .update(`${dataset.name}:${Date.now()}:${process.pid}`)
    .digest('hex')
    .slice(0, 8);
  return `${new Date().toISOString().replace(/[:.]/g, '-')}-${dataset.name}-${suffix}`;
}

function renderSummary(artifact: BenchmarkArtifact): void {
  console.log(
    `\n${artifact.dataset.name} (${artifact.dataset.documents} memories, ` +
    `${artifact.dataset.queries} queries, ${artifact.dataset.granularity})`,
  );
  console.log(`Nemotron: ${artifact.environment.serviceUrl}`);
  console.table(artifact.policies.map(policy => ({
    policy: policy.policy,
    'Recall-all@5': `${((policy.metrics.recallAllAtK['5'] ?? 0) * 100).toFixed(1)}%`,
    'nDCG@5': (policy.metrics.ndcgAtK['5'] ?? 0).toFixed(3),
    MRR: policy.metrics.mrr.toFixed(3),
    abstention: `${(policy.metrics.abstentionAccuracy * 100).toFixed(1)}%`,
    'forbidden/stale': `${(policy.metrics.forbiddenSelectionRate * 100).toFixed(1)}%`,
    'mean selected': policy.metrics.meanSelected.toFixed(2),
    'p95 ms': policy.metrics.latencyMs.p95.toFixed(2),
  })));
}

export async function runRetrievalBenchmark(
  dataset: BenchmarkDataset,
  options: BenchmarkRunOptions,
): Promise<{ artifact: BenchmarkArtifact; artifactPath: string | null }> {
  const config = {
    minimumSimilarity: options.minimumSimilarity ?? 0.18,
    relativeBand: options.relativeBand ?? 0.24,
    limit: options.limit ?? 5,
    maxPerSession: options.maxPerSession ?? 2,
    ks: options.ks ?? (dataset.granularity === 'turn' ? [1, 5, 10, 50] : [1, 5, 10]),
  };
  const client = new EmbeddingClient(options.serviceUrl, options.cachePath);
  try {
    const health = await client.health();
    if (!health) {
      throw new Error(`Nemotron service is unavailable at ${options.serviceUrl}`);
    }
    const queryVectors = await client.embed(
      dataset.queries.map(query => query.text),
      'query',
    );
    const documentVectors = await client.embed(
      dataset.documents.map(document => document.text),
      'passage',
    );
    const documentVectorById = new Map(
      dataset.documents.map((document, index) => [document.id, documentVectors[index]]),
    );
    const documentById = new Map(dataset.documents.map(document => [document.id, document]));
    const policies = buildPolicies(config);
    const policyRuns: PolicyRun[] = [];

    for (const policy of policies) {
      const selectionMap = new Map<string, RankedDocument[]>();
      const latencies = new Map<string, number>();
      const selections: PolicyRun['selections'] = [];
      for (let queryIndex = 0; queryIndex < dataset.queries.length; queryIndex += 1) {
        const query = dataset.queries[queryIndex];
        const candidateDocuments = query.candidateIds
          ? query.candidateIds.map(id => {
              const document = documentById.get(id);
              if (!document) throw new Error(`Missing candidate document ${id}`);
              return document;
            })
          : dataset.documents;
        const candidateVectors = candidateDocuments.map(document => {
          const vector = documentVectorById.get(document.id);
          if (!vector) throw new Error(`Missing vector for ${document.id}`);
          return vector;
        });
        const started = performance.now();
        const ranked = rankByDotProduct(
          candidateDocuments,
          queryVectors[queryIndex],
          candidateVectors,
        );
        const result = policy.select(ranked, query);
        const latencyMs = performance.now() - started;
        selectionMap.set(query.id, result.candidates);
        latencies.set(query.id, latencyMs);
        selections.push({
          queryId: query.id,
          category: query.category,
          expected: query.expected,
          forbidden: query.forbidden ?? [],
          selected: result.candidates.map(document => ({
            id: document.id,
            similarity: Number(document.similarity.toFixed(6)),
          })),
          effectiveMinimumSimilarity: result.effectiveMinimumSimilarity === null
            ? null
            : Number(result.effectiveMinimumSimilarity.toFixed(6)),
          latencyMs: Number(latencyMs.toFixed(4)),
        });
      }
      policyRuns.push({
        policy: policy.name,
        description: policy.description,
        metrics: calculateRetrievalMetrics(
          dataset.queries,
          selectionMap,
          latencies,
          config.ks,
        ),
        selections,
      });
    }

    const artifact: BenchmarkArtifact = {
      schemaVersion: 1,
      runId: runId(dataset),
      createdAt: new Date().toISOString(),
      dataset: {
        name: dataset.name,
        version: dataset.version,
        source: dataset.source,
        granularity: dataset.granularity,
        documents: dataset.documents.length,
        queries: dataset.queries.length,
        sha256: datasetSha256(dataset),
        metadata: dataset.metadata,
      },
      environment: {
        ...gitMetadata(),
        platform: process.platform,
        architecture: process.arch,
        bunVersion: Bun.version,
        serviceUrl: options.serviceUrl,
        embeddingService: health,
        processMemoryBytes: process.memoryUsage(),
      },
      config,
      policies: policyRuns,
    };
    renderSummary(artifact);

    let artifactPath: string | null = null;
    if (options.writeArtifact ?? true) {
      mkdirSync(options.runsDir, { recursive: true });
      artifactPath = join(options.runsDir, `${artifact.runId}.json`);
      writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
      console.log(`Artifact: ${artifactPath}`);
    }
    return { artifact, artifactPath };
  } finally {
    client.close();
  }
}
