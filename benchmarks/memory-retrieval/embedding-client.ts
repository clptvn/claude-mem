import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { Database } from 'bun:sqlite';

type InputType = 'query' | 'passage';

interface EmbeddingResponse {
  model: string;
  data: Array<{ index: number; embedding: number[] }>;
}

interface CacheRow {
  dimensions: number;
  vector: Uint8Array;
}

function sha256(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function vectorToBytes(vector: number[]): Uint8Array {
  return new Uint8Array(new Float32Array(vector).buffer);
}

function bytesToVector(bytes: Uint8Array, dimensions: number): number[] {
  const copy = Uint8Array.from(bytes);
  const floats = new Float32Array(copy.buffer, copy.byteOffset, dimensions);
  return Array.from(floats);
}

export class EmbeddingClient implements Disposable {
  private readonly database: Database;
  private namespace: string;
  private healthCache: Record<string, unknown> | null | undefined;

  constructor(
    readonly serviceUrl: string,
    cachePath: string,
  ) {
    mkdirSync(dirname(cachePath), { recursive: true });
    this.database = new Database(cachePath, { create: true, strict: true });
    this.database.exec('PRAGMA journal_mode = WAL');
    this.database.exec('PRAGMA synchronous = NORMAL');
    const migration = readFileSync(
      join(import.meta.dir, 'migrations', '001_create_embedding_cache.sql'),
      'utf8',
    );
    this.database.exec(migration);
    this.namespace = `${serviceUrl}|unknown`;
  }

  [Symbol.dispose](): void {
    this.database.close();
  }

  close(): void {
    this.database.close();
  }

  async health(): Promise<Record<string, unknown> | null> {
    if (this.healthCache !== undefined) return this.healthCache;
    try {
      const response = await fetch(`${this.serviceUrl}/healthz`, {
        signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok) {
        this.healthCache = null;
        return null;
      }
      const health = await response.json() as Record<string, unknown>;
      const model = health.model as Record<string, unknown> | undefined;
      this.namespace = [
        this.serviceUrl,
        String(model?.model ?? 'unknown'),
        String(model?.embedding_dimensions ?? 'unknown'),
        String(model?.max_sequence_length ?? 'unknown'),
      ].join('|');
      this.healthCache = health;
      return health;
    } catch {
      this.healthCache = null;
      return null;
    }
  }

  async embed(
    inputs: string[],
    inputType: InputType,
    batchSize = 32,
  ): Promise<number[][]> {
    await this.health();
    const vectors: Array<number[] | null> = new Array(inputs.length).fill(null);
    const misses: Array<{ index: number; text: string; hash: string }> = [];
    const getCached = this.database.query<CacheRow, [string, InputType, string]>(
      `SELECT dimensions, vector
       FROM embedding_cache
       WHERE namespace = ? AND input_type = ? AND text_sha256 = ?`,
    );

    for (let index = 0; index < inputs.length; index += 1) {
      const hash = sha256(inputs[index]);
      const cached = getCached.get(this.namespace, inputType, hash);
      if (cached) {
        vectors[index] = bytesToVector(cached.vector, cached.dimensions);
      } else {
        misses.push({ index, text: inputs[index], hash });
      }
    }

    const putCached = this.database.query(
      `INSERT OR REPLACE INTO embedding_cache
         (namespace, input_type, text_sha256, dimensions, vector, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
    );
    for (let offset = 0; offset < misses.length; offset += batchSize) {
      const batch = misses.slice(offset, offset + batchSize);
      const response = await fetch(`${this.serviceUrl}/v1/embeddings`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          input: batch.map(item => item.text),
          input_type: inputType,
        }),
        signal: AbortSignal.timeout(180_000),
      });
      if (!response.ok) {
        throw new Error(
          `Nemotron embedding request failed (${response.status}): ${await response.text()}`,
        );
      }
      const payload = await response.json() as EmbeddingResponse;
      const responseVectors = payload.data
        .sort((left, right) => left.index - right.index)
        .map(item => item.embedding);
      if (responseVectors.length !== batch.length) {
        throw new Error(
          `Nemotron returned ${responseVectors.length} vectors for ${batch.length} inputs`,
        );
      }
      const createdAt = new Date().toISOString();
      this.database.transaction(() => {
        for (let index = 0; index < batch.length; index += 1) {
          const miss = batch[index];
          const vector = responseVectors[index];
          vectors[miss.index] = vector;
          putCached.run(
            this.namespace,
            inputType,
            miss.hash,
            vector.length,
            vectorToBytes(vector),
            createdAt,
          );
        }
      })();
    }

    if (vectors.some(vector => vector === null)) {
      throw new Error('Embedding cache returned an incomplete vector set');
    }
    return vectors as number[][];
  }
}
