"""Download Wikipedia rows and sync them to the configured catalog and schema."""

import asyncio
import os
import ssl
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import certifi
import httpx
import wikipediaapi

if __package__:
    from src import config
    from src.topics import wiki_topics
else:
    import config
    from topics import wiki_topics

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

topics = wiki_topics


class WikipediaDocument(TypedDict):
    pageid: int
    title: str
    texts: str
    topic: str
    summary: str
    fullurl: str
    categories: list[str]
    images: list[str]
    created_at: datetime | None
    updated_at: datetime | None


async def load_wikipedia_pages_async(
    max_docs_per_topic: int = 10,
    *,
    concurrency: int = 4,
    user_agent: str = "DrillingResearch/1.0 (mai.la@berkeley.edu)",
    language: str = "en",
    ca_file: str | Path | None = None,
) -> list[WikipediaDocument]:
    """Search topics and return unique pages as plain dictionaries.

    Each topic contributes at most max_docs_per_topic search results; overlapping
    pages are returned once, in topic/search order, with the first matching topic.
    Missing pages are skipped. created_at and updated_at are timezone-aware UTC
    datetime objects from the earliest and latest available revisions, or None
    if unavailable.
    Categories and images contain Wikipedia category/file titles (not objects or
    downloaded image bytes). Text is the full extract, without truncation.
    """
    if not 1 <= max_docs_per_topic <= 500:
        raise ValueError("max_docs_per_topic must be between 1 and 500")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ca_file = ca_file or os.environ.get("SSL_CERT_FILE")
    if ca_file:
        ssl_context.load_verify_locations(cafile=str(Path(ca_file).expanduser()))

    wiki = wikipediaapi.AsyncWikipedia(
        user_agent=user_agent,
        language=language,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        transport=httpx.AsyncHTTPTransport(verify=ssl_context),
        timeout=30.0,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def revision_timestamp(pageid: int, direction: str) -> datetime | None:
        # Wikipedia-API 0.16 has no public revisions method. Reuse its async
        # request layer for the MediaWiki revisions query, including retries/TLS.
        result = await wiki._get(
            language=wiki.language,
            params={
                "action": "query",
                "format": "json",
                "prop": "revisions",
                "pageids": str(pageid),
                "rvprop": "timestamp",
                "rvlimit": 1,
                "rvdir": direction,
            },
        )
        page_data = result.get("query", {}).get("pages", {}).get(str(pageid), {})
        revisions = page_data.get("revisions", [])
        timestamp = revisions[0].get("timestamp") if revisions else None
        # MediaWiki already supplies UTC ISO 8601 timestamps ending in Z.
        return datetime.fromisoformat(timestamp) if timestamp else None

    async def download(
        page: wikipediaapi.AsyncWikipediaPage, topic: str
    ) -> WikipediaDocument | None:
        async with semaphore:
            if not await page.exists():
                return None
            # Await related lazy properties sequentially to reuse their caches.
            pageid = int(await page.pageid)
            texts = await page.text
            summary = await page.summary
            fullurl = await page.fullurl
            categories = await page.categories
            images = await page.images
            created_at = await revision_timestamp(pageid, "newer")
            updated_at = await revision_timestamp(pageid, "older")
            return {
                "pageid": pageid,
                "title": page.title,
                "texts": texts,
                "topic": topic,
                "summary": summary,
                "fullurl": fullurl,
                "categories": sorted(categories),
                "images": sorted(images),
                "created_at": created_at,
                "updated_at": updated_at,
            }

    rows: list[WikipediaDocument] = []
    seen_titles: set[str] = set()
    seen_pageids: set[int] = set()
    try:
        for topic in dict.fromkeys(topics):
            results = await wiki.search(topic, limit=max_docs_per_topic)
            tasks = []
            # TaskGroup cancels and awaits remaining downloads if one fails.
            async with asyncio.TaskGroup() as group:
                for page in list(results.pages.values())[:max_docs_per_topic]:
                    if page.title not in seen_titles:
                        seen_titles.add(page.title)
                        tasks.append(group.create_task(download(page, topic)))
            for task in tasks:
                row = task.result()
                if row is not None and row["pageid"] not in seen_pageids:
                    seen_pageids.add(row["pageid"])
                    rows.append(row)
    finally:
        # Wikipedia-API 0.16 has no public close/aclose method.
        await wiki._client.aclose()
    return rows


def load_wikipedia_pages(
    max_docs_per_topic: int = 10,
    *,
    concurrency: int = 3,
    user_agent: str = "DrillingResearch/1.0 (mai.la@berkeley.edu)",
    language: str = "en",
    ca_file: str | Path | None = None,
) -> list[WikipediaDocument]:
    """Run the async loader with asyncio.run() and return Spark-ready rows.

    Example::

        rows = load_wikipedia_pages()

    In Jupyter/Databricks notebooks use await load_wikipedia_pages_async(...)
    because asyncio.run() cannot run inside an already running event loop.
    """
    return asyncio.run(
        load_wikipedia_pages_async(
            max_docs_per_topic,
            concurrency=concurrency,
            user_agent=user_agent,
            language=language,
            ca_file=ca_file,
        )
    )


async def sync_wikipedia_pages(
    spark: "SparkSession",
    catalog: str | None = None,
    schema: str | None = None,
    table: str = "wiki_pages",
) -> "DataFrame":
    """Load Wikipedia pages and append to a Delta table, creating it if absent.

    In Databricks/Jupyter: df = await sync_wikipedia_pages(spark).
    In a synchronous script: df = asyncio.run(sync_wikipedia_pages(spark)).
    Repeated calls append all loaded pages, including any already stored pages.
    """
    from pyspark.sql.types import (
        ArrayType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    if catalog is None:
        catalog = config.wiki_config["CATALOG"]
    if schema is None:
        schema = config.wiki_config["SCHEMA"]
    table_name = ".".join(
        f"`{part.replace('`', '``')}`" for part in (catalog, schema, table)
    )
    rows = await load_wikipedia_pages_async()
    spark_schema = StructType([
        StructField("pageid", LongType()),
        StructField("title", StringType()),
        StructField("texts", StringType()),
        StructField("topic", StringType()),
        StructField("summary", StringType()),
        StructField("fullurl", StringType()),
        StructField("categories", ArrayType(StringType())),
        StructField("images", ArrayType(StringType())),
        StructField("created_at", TimestampType()),
        StructField("updated_at", TimestampType()),
    ])
    dataframe = spark.createDataFrame(rows, schema=spark_schema)
    mode = "append" if spark.catalog.tableExists(table_name) else "overwrite"
    schema_option = "mergeSchema" if mode == "append" else "overwriteSchema"
    dataframe.write.format("delta").mode(mode).option(
        schema_option, "true"
    ).saveAsTable(table_name)
    return dataframe


# Run these lines in a separate Databricks notebook cell after loading this file:
# df = await sync_wikipedia_pages(spark)
# df.show(5)

# To inspect downloaded rows without writing to the table, run instead:
# rows = await load_wikipedia_pages_async()
# print(rows[:1])
