CREATE TABLE IF NOT EXISTS embedding_cache (
  namespace TEXT NOT NULL,
  input_type TEXT NOT NULL CHECK (input_type IN ('query', 'passage')),
  text_sha256 TEXT NOT NULL,
  dimensions INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (namespace, input_type, text_sha256)
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_created_at
  ON embedding_cache(created_at);
