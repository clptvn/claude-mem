from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nemotron_memory_service.vector_store import NemotronVectorStore


class FakeBatcher:
    async def submit(self, texts: list[str], input_type: str) -> list[list[float]]:
        if input_type == "query":
            return [[1.0, 0.0] for _ in texts]
        return [
            [1.0, 0.0] if "memory" in text.lower() else [0.0, 1.0]
            for text in texts
        ]


class VectorStoreTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = NemotronVectorStore(
            Path(self.temp_dir.name),
            FakeBatcher(),  # type: ignore[arg-type]
        )
        await self.store.call(
            "chroma_create_collection", {"collection_name": "cm__test"}
        )

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_explicit_embeddings_rank_memory_document_first(self) -> None:
        self.assertFalse(self.store.client.get_settings().anonymized_telemetry)

        await self.store.call(
            "chroma_add_documents",
            {
                "collection_name": "cm__test",
                "ids": ["obs_1_narrative", "obs_2_narrative"],
                "documents": [
                    "One shared memory model serves Claude and Codex.",
                    "Bread uses flour and water.",
                ],
                "metadatas": [
                    {"sqlite_id": 1, "doc_type": "observation"},
                    {"sqlite_id": 2, "doc_type": "observation"},
                ],
            },
        )

        result = await self.store.call(
            "chroma_query_documents",
            {
                "collection_name": "cm__test",
                "query_texts": ["How do agents share memory?"],
                "n_results": 2,
                "include": ["documents", "metadatas", "distances"],
            },
        )

        self.assertEqual(result["ids"][0][0], "obs_1_narrative")  # type: ignore[index]
        self.assertLess(
            result["distances"][0][0],  # type: ignore[index]
            result["distances"][0][1],  # type: ignore[index]
        )


if __name__ == "__main__":
    unittest.main()
