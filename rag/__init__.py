"""RAG from scratch: ingestion, chunking, embeddings, search, answers."""

from .chunking import chunk_document, split_into_chunks, split_sentences
from .document import Chunk, Document
from .embeddings import (
    EmbeddedChunk,
    Embedder,
    HashingEmbedder,
    cosine_similarity,
    embed_chunks,
    normalize,
)
from .hybrid import BM25Index, HybridRetriever, reciprocal_rank_fusion, tokenize
from .ingest import ingest_directory, load_document
from .pipeline import build_chunks
from .retrieval import Retriever, matches_metadata, mmr_rerank
from .vector_store import SearchResult, VectorStore

__all__ = [
    "BM25Index",
    "Chunk",
    "Document",
    "EmbeddedChunk",
    "Embedder",
    "HashingEmbedder",
    "HybridRetriever",
    "Retriever",
    "SearchResult",
    "VectorStore",
    "build_chunks",
    "chunk_document",
    "cosine_similarity",
    "embed_chunks",
    "ingest_directory",
    "load_document",
    "matches_metadata",
    "mmr_rerank",
    "normalize",
    "reciprocal_rank_fusion",
    "split_into_chunks",
    "split_sentences",
    "tokenize",
]
