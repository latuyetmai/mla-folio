"""Configuration for the Wikipedia assistant."""

import os

CATALOG = os.getenv("CATALOG", "ml_dev")
SCHEMA = os.getenv("SCHEMA", "wiki")
VECTORSEARCH_ENDPOINT = os.getenv("VECTORSEARCH_ENDPOINT", "databricks-ai-search")

wiki_config = {
    "CATALOG": CATALOG,
    "SCHEMA": SCHEMA,
    "MODEL_VERSION": "v0.0.1",
    "EMBEDDING_MODEL": "databricks-bge-large-en",
    "LLM_ENDPOINT_NAME": "databricks-meta-llama-3-3-70b-instruct",
    "LLM_FALLBACK_ENDPOINT_NAME": "databricks-qwen35-122b-a10b",
    "LLM_FALLBACK_ENDPOINT_NAME2": "databricks-gpt-oss-120b",
    "CLASSIFICATION_LLM_ENDPOINT_NAME": "databricks-gemma-3-12b",
    "SUMMARIZATION_LLM_ENDPOINT_NAME": "databricks-gpt-oss-120b",
    "VECTORSEARCH_ENDPOINT": VECTORSEARCH_ENDPOINT,
    "VECTORSEARCH_WIKI_INDEX": f"{CATALOG}.{SCHEMA}.wiki_pages_chunks_index",
}
