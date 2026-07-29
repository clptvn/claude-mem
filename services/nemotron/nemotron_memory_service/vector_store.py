from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .model import EmbeddingBatcher


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


class NemotronVectorStore:
    WRITE_TOOLS = {
        "chroma_add_documents",
        "chroma_update_documents",
        "chroma_delete_documents",
        "chroma_delete_collection",
        "chroma_modify_collection",
    }

    def __init__(self, path: Path, batcher: EmbeddingBatcher):
        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.batcher = batcher
        self._store_lock = asyncio.Lock()

    async def call(self, tool_name: str, arguments: dict) -> object:
        handlers = {
            "chroma_list_collections": self._list_collections,
            "chroma_create_collection": self._create_collection,
            "chroma_get_collection_count": self._get_collection_count,
            "chroma_get_documents": self._get_documents,
            "chroma_query_documents": self._query_documents,
            "chroma_add_documents": self._add_documents,
            "chroma_update_documents": self._update_documents,
            "chroma_delete_documents": self._delete_documents,
            "chroma_delete_collection": self._delete_collection,
            "chroma_modify_collection": self._modify_collection,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"Unsupported vector tool: {tool_name}")
        return await handler(arguments)

    def _collection(self, name: str):
        return self.client.get_collection(name=name, embedding_function=None)

    async def _list_collections(self, args: dict) -> list[str]:
        async with self._store_lock:
            collections = await asyncio.to_thread(
                self.client.list_collections,
                limit=args.get("limit"),
                offset=args.get("offset"),
            )
        return [collection.name for collection in collections] or [
            "__NO_COLLECTIONS_FOUND__"
        ]

    async def _create_collection(self, args: dict) -> str:
        name = args["collection_name"]
        async with self._store_lock:
            try:
                await asyncio.to_thread(
                    self.client.create_collection,
                    name=name,
                    embedding_function=None,
                    configuration={"hnsw": {"space": "cosine"}},
                    metadata=args.get("metadata"),
                )
            except Exception as exc:
                if "already exists" in str(exc).lower():
                    raise RuntimeError(f"Collection {name} already exists") from exc
                raise
        return f"Successfully created collection {name} with Nemotron embeddings"

    async def _get_collection_count(self, args: dict) -> int:
        async with self._store_lock:
            collection = self._collection(args["collection_name"])
            return await asyncio.to_thread(collection.count)

    async def _get_documents(self, args: dict) -> dict:
        kwargs = {
            key: args[key]
            for key in ("ids", "where", "where_document", "include", "limit", "offset")
            if key in args and args[key] is not None
        }
        async with self._store_lock:
            collection = self._collection(args["collection_name"])
            result = await asyncio.to_thread(collection.get, **kwargs)
        return _json_safe(result)

    async def _query_documents(self, args: dict) -> dict:
        query_texts = args.get("query_texts") or []
        if not query_texts:
            raise ValueError("query_texts cannot be empty")
        embeddings = await self.batcher.submit(query_texts, "query")
        kwargs = {
            "query_embeddings": embeddings,
            "n_results": args.get("n_results", 5),
        }
        for key in ("where", "where_document", "include"):
            if key in args and args[key] is not None:
                kwargs[key] = args[key]
        async with self._store_lock:
            collection = self._collection(args["collection_name"])
            result = await asyncio.to_thread(collection.query, **kwargs)
        return _json_safe(result)

    async def _add_documents(self, args: dict) -> str:
        documents = args.get("documents") or []
        ids = args.get("ids") or []
        if not documents or len(documents) != len(ids):
            raise ValueError("documents and ids must be non-empty and have equal lengths")
        embeddings = await self.batcher.submit(documents, "passage")
        kwargs = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
        }
        if args.get("metadatas") is not None:
            kwargs["metadatas"] = args["metadatas"]
        async with self._store_lock:
            collection = self._collection(args["collection_name"])
            await asyncio.to_thread(collection.add, **kwargs)
        return (
            f"Successfully added {len(documents)} documents to collection "
            f"{args['collection_name']}"
        )

    async def _update_documents(self, args: dict) -> str:
        ids = args.get("ids") or []
        if not ids:
            raise ValueError("ids cannot be empty")
        kwargs: dict[str, object] = {"ids": ids}
        documents = args.get("documents")
        if documents is not None:
            if len(documents) != len(ids):
                raise ValueError("documents and ids must have equal lengths")
            kwargs["documents"] = documents
            kwargs["embeddings"] = await self.batcher.submit(documents, "passage")
        if args.get("metadatas") is not None:
            kwargs["metadatas"] = args["metadatas"]
        if args.get("embeddings") is not None:
            kwargs["embeddings"] = args["embeddings"]
        async with self._store_lock:
            collection = self._collection(args["collection_name"])
            await asyncio.to_thread(collection.update, **kwargs)
        return (
            f"Successfully updated {len(ids)} documents in collection "
            f"{args['collection_name']}"
        )

    async def _delete_documents(self, args: dict) -> str:
        kwargs = {
            key: args[key]
            for key in ("ids", "where", "where_document")
            if key in args and args[key] is not None
        }
        async with self._store_lock:
            collection = self._collection(args["collection_name"])
            await asyncio.to_thread(collection.delete, **kwargs)
        return f"Successfully deleted documents from {args['collection_name']}"

    async def _delete_collection(self, args: dict) -> str:
        name = args["collection_name"]
        async with self._store_lock:
            await asyncio.to_thread(self.client.delete_collection, name=name)
        return f"Successfully deleted collection {name}"

    async def _modify_collection(self, args: dict) -> str:
        kwargs = {}
        if args.get("new_name") is not None:
            kwargs["name"] = args["new_name"]
        if args.get("new_metadata") is not None:
            kwargs["metadata"] = args["new_metadata"]
        async with self._store_lock:
            collection = self._collection(args["collection_name"])
            await asyncio.to_thread(collection.modify, **kwargs)
        return f"Successfully modified collection {args['collection_name']}"
