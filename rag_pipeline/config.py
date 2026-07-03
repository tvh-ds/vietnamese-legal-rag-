"""Configuration loader for the chunking + retrieval pipeline."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _resolve_env(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable value."""
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{(\w+)\}")
    return pattern.sub(lambda m: os.environ.get(m.group(1), ""), value)


@dataclass
class ChunkingConfig:
    """Chunking parameters (Sections 3 & 4 of Pipeline.md)."""

    max_tokens: int = 512
    min_tokens: int = 30
    overlap_tokens: int = 0


@dataclass
class EmbeddingAPIConfig:
    """Company embedding API parameters."""

    model: str = "Vietnamese_Embedding"
    max_seq_length: int = 2048
    batch_size: int = 64
    timeout_sec: int = 60


@dataclass
class EmbeddingConfig:
    """Embedding model parameters (Section 7 of Pipeline.md)."""

    backend: str = "local"  # "local" or "api"
    model_name: str = "AITeamVN/Vietnamese_Embedding"
    batch_size: int = 16
    device: str = "cpu"
    api: EmbeddingAPIConfig = field(default_factory=EmbeddingAPIConfig)


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
    fusion_weight: float = 0.5
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
class LLMConfig:
    """LLM API parameters for context generation."""

    model: str = "gemma-4-31B-it"
    temperature: float = 0.3
    max_output_tokens: int = 120
    timeout_sec: int = 30


@dataclass
class ContextConfig:
    """Context generation parameters (Section 5 of Pipeline.md)."""

    enabled: bool = True
    use_llm: bool = False
    max_context_tokens: int = 100
    llm: LLMConfig = field(default_factory=LLMConfig)


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
        """Load configuration from a YAML file. Missing keys fall back to defaults."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        emb_raw = raw.get("embedding", {})
        emb_api_raw = emb_raw.pop("api", {}) if isinstance(emb_raw, dict) else {}
        ctx_raw = raw.get("context", {})
        llm_raw = ctx_raw.pop("llm", {}) if isinstance(ctx_raw, dict) else {}

        # Shared API config (base_url + api_key)
        shared_api = raw.get("api", {})
        base_url = shared_api.get("base_url", "").rstrip("/")
        api_key = shared_api.get("api_key", "")

        # Attach shared API credentials to the config
        result = cls(
            chunking=ChunkingConfig(**raw.get("chunking", {})),
            embedding=EmbeddingConfig(
                **emb_raw,
                api=EmbeddingAPIConfig(
                    model=emb_api_raw.get("model", "Vietnamese_Embedding"),
                    max_seq_length=emb_api_raw.get("max_seq_length", 2048),
                    batch_size=emb_api_raw.get("batch_size", 64),
                    timeout_sec=emb_api_raw.get("timeout_sec", 60),
                ),
            ),
            vector_store=VectorStoreConfig(**raw.get("vector_store", {})),
            query_rewriter=QueryRewriterConfig(**raw.get("query_rewriter", {})),
            bm25=BM25Config(**raw.get("bm25", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            reranker=RerankerConfig(**raw.get("reranker", {})),
            context=ContextConfig(
                **ctx_raw,
                llm=LLMConfig(
                    model=llm_raw.get("model", "gemma-4-31B-it"),
                    temperature=llm_raw.get("temperature", 0.3),
                    max_output_tokens=llm_raw.get("max_output_tokens", 120),
                    timeout_sec=llm_raw.get("timeout_sec", 30),
                ),
            ),
            data=DataConfig(**raw.get("data", {})),
            metadata=MetadataConfig(**raw.get("metadata", {})),
        )
        result._api_base_url = base_url
        result._api_key = api_key
        return result
        config._api_base_url = base_url
        config._api_key = api_key
        return config


def load_config(path: str | Path = "config.yaml") -> Config:
    """Convenience loader. Resolves the path against cwd."""
    config_path = Path(path)
    if not config_path.is_absolute():
        package_dir = Path(__file__).parent
        candidate = package_dir / config_path
        if candidate.exists():
            config_path = candidate
    return Config.from_yaml(str(config_path))
