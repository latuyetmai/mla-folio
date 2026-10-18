"""Split Wikipedia text into word-limited chunks for indexing."""

import re
from typing import TYPE_CHECKING

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.functions import col, concat_ws, pandas_udf, posexplode
from pyspark.sql.types import ArrayType, StringType, TimestampNTZType, TimestampType

if __package__:
    from . import config
else:
    import config

if TYPE_CHECKING:
    from databricks.ai_search.client import AISearchClient
    from databricks.ai_search.index import AISearchIndex
    from pyspark.sql import DataFrame, SparkSession


def _validate_chunk_options(max_length: int, overlap: int) -> None:
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < max_length:
        raise ValueError("overlap must be an integer between 0 and max_length - 1")


def recursive_split_text(
    text: str | None,
    max_length: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text using paragraphs, newlines, sentences, then spaces.

    Lengths count whitespace-separated words, rather than characters or tokens.
    Overlap retains up to the requested number of words where split boundaries
    allow it. Null, empty, and whitespace-only inputs return an empty list.
    """
    _validate_chunk_options(max_length, overlap)
    if text is None or not text.strip():
        return []

    # Normalize tabs and other whitespace so the final space separator can
    # always enforce the word limit, while retaining paragraph/newline breaks.
    text = re.sub(r"[^\S\n]+", " ", text.replace("\r\n", "\n"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=overlap,
        length_function=lambda value: len(value.split()),
        separators=["\n\n", "\n", ".", " "],
        keep_separator="end",
        is_separator_regex=False,
    )
    return splitter.split_text(text)


def create_recursive_split_udf(max_length: int = 500, overlap: int = 50):
    """Build a Pandas UDF with configurable word limits and overlap."""
    _validate_chunk_options(max_length, overlap)

    @pandas_udf(ArrayType(StringType()))
    def split_batch(texts: pd.Series) -> pd.Series:
        return texts.apply(
            lambda text: recursive_split_text(
                None if pd.isna(text) else text,
                max_length=max_length,
                overlap=overlap,
            )
        )

    return split_batch


recursive_split_udf = create_recursive_split_udf(max_length = 500, overlap = 50)


def load_wikipedia_chunks(
    spark: "SparkSession",
    table_name: str | None = None,
) -> "DataFrame":
    """Return one row per chunk, ready for writing to an indexing source table.

    Chunk indexes are zero-based. IDs use the form pageid_chunk_index; source
    pageids must be unique for the resulting chunk_ids to be unique. Pages with
    no chunks produce no rows. Timestamp columns retain their original types;
    all other columns are cast to strings. This function does not write a table.
    """
    if table_name is None:
        table_name = (
            f"{config.wiki_config['CATALOG']}.{config.wiki_config['SCHEMA']}.wiki_pages"
        )
    pages = spark.table(table_name)
    chunks = pages.withColumn("all_chunks", recursive_split_udf("texts"))
    chunks = chunks.select(
        "*", posexplode("all_chunks").alias("chunk_index", "text_chunks")
    )
    chunks = chunks.withColumn(
        "chunk_id",
        concat_ws(
            "_", col("pageid").cast(StringType()), col("chunk_index").cast(StringType())
        ),
    ).drop("all_chunks", "chunk_index", "texts")

    return chunks.select(
        *[
            col(field.name)
            if isinstance(field.dataType, (TimestampType, TimestampNTZType))
            else col(field.name).cast(StringType()).alias(field.name)
            for field in chunks.schema.fields
        ]
    )


def sync_wikipedia_chunks(
    spark: "SparkSession",
    catalog: str | None = None,
    schema: str | None = None,
    table: str = "wiki_pages_chunks",
) -> "DataFrame":
    """Create or upsert the chunk table with Change Data Feed enabled.

    Return chunked_pages with one row per chunk_id. If the source contains
    duplicate IDs, dropDuplicates retains one row per ID; which row survives is
    unspecified. Null pageids or chunk_ids are rejected before writing.
    Existing target rows absent from the source are retained.
    """
    if catalog is None:
        catalog = config.wiki_config["CATALOG"]
    if schema is None:
        schema = config.wiki_config["SCHEMA"]
    chunked_pages = load_wikipedia_chunks(
        spark, table_name=f"{catalog}.{schema}.wiki_pages"
    )
    if chunked_pages.filter(
        col("pageid").isNull() | col("chunk_id").isNull()
    ).limit(1).count():
        raise ValueError("Every chunk must have a non-null pageid and chunk_id")
    chunked_pages = chunked_pages.dropDuplicates(["chunk_id"])

    table_name = ".".join(
        f"`{part.replace('`', '``')}`" for part in (catalog, schema, table)
    )
    if not spark.catalog.tableExists(table_name):
        (
            chunked_pages.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .option("delta.enableChangeDataFeed", "true")
            .saveAsTable(table_name)
        )
    else:
        from delta.tables import DeltaTable

        spark.sql(
            f"ALTER TABLE {table_name} SET TBLPROPERTIES "
            "('delta.enableChangeDataFeed' = 'true')"
        )
        (
            DeltaTable.forName(spark, table_name)
            .alias("target")
            .merge(chunked_pages.alias("source"), "target.chunk_id = source.chunk_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    return chunked_pages


def create_vector_index(
    spark: "SparkSession",
    source_table_name: str | None = None,
    *,
    client: "AISearchClient | None" = None,
) -> "AISearchIndex":
    """Create a TRIGGERED Delta Sync index, or sync it if it already exists.

    Read endpoint/model settings from config.wiki_config. The index name is
    the fully qualified source table name plus _index. All source columns are
    included. Return the index handle; creation/sync can continue asynchronously.
    """
    from databricks.ai_search.client import AISearchClient
    import time

    if source_table_name is None:
        source_table_name = (
            f"{config.wiki_config['CATALOG']}.{config.wiki_config['SCHEMA']}.wiki_pages_chunks"
        )
    endpoint_name = config.wiki_config.get("VECTORSEARCH_ENDPOINT")
    embedding_model = config.wiki_config.get("EMBEDDING_MODEL")
    if not endpoint_name:
        raise ValueError("wiki_config must define VECTORSEARCH_ENDPOINT")
    if not embedding_model:
        raise ValueError("wiki_config must define EMBEDDING_MODEL")

    if client is None:
        client = AISearchClient()
    index_name = f"{source_table_name}_index"
    if client.index_exists(endpoint_name=endpoint_name, index_name=index_name):
        index = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
        index.sync()
        print(index.describe())
        return index

    columns_to_sync = spark.table(source_table_name).columns
    if not {"chunk_id", "text_chunks"}.issubset(columns_to_sync):
        raise ValueError("The source table must contain chunk_id and text_chunks")
    index = client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        source_table_name=source_table_name,
        index_name=index_name,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="text_chunks",
        embedding_model_endpoint_name=embedding_model,
        columns_to_sync=columns_to_sync,
    )
    print(index.describe())
    while not index.describe().get("status")["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(30)
    return index

# Databricks notebook usage (synchronous; no await needed):
# chunked_pages = sync_wikipedia_chunks(spark)
# chunked_pages.show(5)

# Create or sync the vector index for the chunked pages:
if __name__ == "__main__":
    vector_index = create_vector_index(spark)
