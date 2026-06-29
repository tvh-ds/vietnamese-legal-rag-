"""Configuration loader for the chunking + retrieval pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ChunkingConfig:
    """Chunking parameters (Sections 3 & 4 of Pipeline.md)."""

    max_tokens: int = 512
    min_tokens: int = 30
    overlap_tokens: int = 0


@dataclass
class EmbeddingConfig:
    """Embedding model parameters (Section 7 of Pipeline.md)."""

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class VectorStoreConfig:
    """Vector database parameters (Section 8 of Pipeline.md)."""

    type: str = "chromadb"
    path: str = "./chroma_db"
    collection_name: str = "vbqppl_legal_chunks"


@dataclass
class QueryRewriterConfig:
    """Query rewriting parameters (Section 9 of Pipeline.md)."""

    enabled: bool = True
    expand_abbreviations: bool = True
    append_context: bool = True
    context_phrase: str = "theo pháp luật hiện hành"


@dataclass
class BM25Config:
    """BM25 sparse retrieval parameters."""

    enabled: bool = True
    top_k: int = 20
    k1: float = 1.5
    b: float = 0.75
    index_path: str = "./chroma_db/bm25_index.pkl"


@dataclass
class RetrievalConfig:
    """Retrieval parameters (Section 10 of Pipeline.md)."""

    top_k: int = 10
    similarity_threshold: float = 0.3
    enable_metadata_filtering: bool = True
    enable_graph_expansion: bool = True
    expansion_max_articles: int = 5
    vector_weight: float = 0.7
    metadata_boost: float = 0.2
    graph_boost: float = 0.1


@dataclass
class RerankerConfig:
    """Reranker parameters (Section 11 of Pipeline.md)."""

    enabled: bool = True
    lambda_mmr: float = 0.7
    keyword_boost: float = 0.15
    diversity_weight: float = 0.15


@dataclass
class ContextConfig:
    """Context generation parameters (Section 5 of Pipeline.md)."""

    enabled: bool = True
    use_llm: bool = False
    max_context_tokens: int = 100


@dataclass
class DataConfig:
    """Data source configuration."""

    path: str = "./data"


@dataclass
class MetadataConfig:
    """Metadata filtering preferences."""

    prefer_active: bool = True
    topic_boost: bool = True


@dataclass
class Config:
    """Root configuration aggregating all pipeline sections."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    query_rewriter: QueryRewriterConfig = field(default_factory=QueryRewriterConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    data: DataConfig = field(default_factory=DataConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file.

        Missing keys fall back to dataclass defaults.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            chunking=ChunkingConfig(**raw.get("chunking", {})),
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
            vector_store=VectorStoreConfig(**raw.get("vector_store", {})),
            query_rewriter=QueryRewriterConfig(**raw.get("query_rewriter", {})),
            bm25=BM25Config(**raw.get("bm25", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            reranker=RerankerConfig(**raw.get("reranker", {})),
            context=ContextConfig(**raw.get("context", {})),
            data=DataConfig(**raw.get("data", {})),
            metadata=MetadataConfig(**raw.get("metadata", {})),
        )


def load_config(path: str | Path = "config.yaml") -> Config:
    """Convenience loader. Resolves the path against cwd."""
    config_path = Path(path)
    if not config_path.is_absolute():
        # Try relative to the rag_pipeline package directory
        package_dir = Path(__file__).parent
        candidate = package_dir / config_path
        if candidate.exists():
            config_path = candidate
    return Config.from_yaml(str(config_path))
