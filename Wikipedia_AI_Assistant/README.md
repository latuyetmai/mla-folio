# Wikipedia Drilling Assistant

A RAG assistant for drilling, drilling equipment, and geology questions, built with a selected subset of Wikipedia, Databricks Vector Search, and a LangChain ReAct agent. Answers include Wikipedia source links, and Lakebase checkpoints preserve conversation history.


## How it works

1. Download Wikipedia pages for the topics in `src/topics.py` and store them in a Delta table.
2. Split pages into chunks of up to 500 words with a target 50-word overlap, then create a Delta Sync Vector Search index.
3. For each question, the retrieval tool performs three steps:
   - **Topic prefilter:** Classify the question into allowed topics using `TopicSelection`; discard unknown topics.
   - **Filtered retrieval:** Apply the topic filter to hybrid Vector Search, retrieve 20 results, rerank by summary, and retain 10.
   - **Relevance post-filter:** Classify each chunk concurrently and remove irrelevant context before returning documents to the agent.
4. The agent uses the evidence to answer with citations, or explicitly labels general-knowledge answers when evidence is unavailable.

## Setup

Use Python 3.12 and Poetry. Data preparation requires a Databricks Spark runtime with Delta support. You also need access to a catalog/schema, Vector Search endpoint, Lakebase instance, and the model endpoints configured in `src/config.py`.

```bash
git clone https://github.com/latuyetmai/wikipedia-drilling-assistant.git
cd wikipedia-drilling-assistant
poetry install --no-root --with dev
cp .env.example .env
```

Edit `.env` with your own values:

- `DATABRICKS_HOST`: workspace URL.
- `DATABRICKS_TOKEN`, or `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` for service-principal authentication. Remove unused placeholder credentials.
- `CATALOG`, `SCHEMA`, and `VECTORSEARCH_ENDPOINT`: data and index locations.
- `LAKEBASE_INSTANCE_NAME` and `LAKEBASE_HOST`: an existing Lakebase instance and its matching hostname.
- `MLFLOW_TRACKING_URI=databricks` and `MLFLOW_EXPERIMENT_ID`: for notebook tracing.

Review the LLM and embedding endpoints in `src/config.py` for your workspace. Keep credentials out of version control. OpenAI and Anthropic keys in the example file are not required by the current Databricks model configuration.

## Build the data and index

Run these steps in a Databricks notebook with this repository available on the Python path, dependencies installed, configuration loaded, and a `spark` session:

```python
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import sync_wikipedia_pages
from src.indexing import sync_wikipedia_chunks, create_vector_index

pages = await sync_wikipedia_pages(spark)
chunks = sync_wikipedia_chunks(spark)
index = create_vector_index(spark)
```

This creates `{CATALOG}.{SCHEMA}.wiki_pages`, `wiki_pages_chunks`, and `wiki_pages_chunks_index`. Wait for the index to be ready before querying. Re-running ingestion appends page rows; chunk writes merge by `chunk_id`. Calling `create_vector_index()` on an existing index triggers a sync.

## Run the agent

Open `01_agentic_wikipedia_aimpoint_interview.ipynb` and run the custom cells under **a) GenAI Application Development** after preparing the index. Select the project environment when running locally; the earlier starter cells are a separate FAISS example.

The compiled agent can also be imported directly from the repository root:

```python
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

from src.wiki_agent import wiki_agent

config = {
    "configurable": {"thread_id": f"wiki-{uuid4().hex}"},
    "recursion_limit": 100,
}
response = wiki_agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about horizontal drilling."}]},
    config=config,
)
print(response["messages"][-1].content)

# Reuse the conversation thread for a follow-up.
for update in wiki_agent.stream(
    {"messages": [{"role": "user", "content":
        "What is the difference between horizontal drilling and lateral drilling?"}]},
    config=config,
    stream_mode="updates",
):
    print(update, flush=True)
```

Importing the `wiki_agent` export initializes the model clients and Lakebase saver. Use a new `thread_id` for each conversation. The notebook adds `src/` to its Python path and uses the equivalent `from wiki_agent import wiki_agent` import.

For MLflow traces, call `mlflow.langchain.autolog()` before invocation. The exported graph supports invocation and streaming for additional serving layers; an MCP server or frontend is not included in this workflow.

## Test

Run the test suite from the repository root:

```bash
poetry run python -m pytest tests -q
```

The live retrieval test is skipped by default. To query your configured Databricks index and models:

```bash
WIKI_LIVE_TEST=1 poetry run python -m pytest tests/test_wiki_retrieval_live.py -s
```

Live runs require valid workspace credentials and an initialized index, and use remote resources. The notebook's invoke and stream cells provide end-to-end checks that also exercise Lakebase memory.

## Key files

| Path | Purpose |
| --- | --- |
| `src/config.py`, `src/topics.py` | Configuration and Wikipedia topic scope |
| `src/data_loader.py` | Asynchronous page ingestion and Delta writes |
| `src/indexing.py` | Chunking, Delta upserts, and index creation/sync |
| `src/wiki_agent.py` | Retrieval chain and ReAct agent factory/export |
| `src/lakebase_checkpoint.py` | Persistent checkpoint storage |
| `skills/wiki_agent_prompt.md` | Agent instructions and citation rules |
| `01_agentic_wikipedia_aimpoint_interview.ipynb` | Workflow explanation and runnable examples |
| `tests/` | Retrieval, agent, configuration, and checkpoint tests |
| `my_notes.md` | Development steps and prompts |
