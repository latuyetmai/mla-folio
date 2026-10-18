import asyncio
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import config, data_loader, indexing


@pytest.mark.parametrize("catalog", [None, "wiki_test"])
@pytest.mark.parametrize("schema", [None, "research_test"])
def test_catalog_environment_and_index_name(monkeypatch, catalog, schema):
    if catalog is None:
        monkeypatch.delenv("CATALOG", raising=False)
    else:
        monkeypatch.setenv("CATALOG", catalog)
    if schema is None:
        monkeypatch.delenv("SCHEMA", raising=False)
    else:
        monkeypatch.setenv("SCHEMA", schema)
    settings = runpy.run_path(str(Path(config.__file__)))['wiki_config']
    expected = catalog or "rh_dev"
    expected_schema = schema or "wiki"
    assert settings["CATALOG"] == expected
    assert settings["SCHEMA"] == expected_schema
    assert settings["VECTORSEARCH_WIKI_INDEX"] == (
        f"{expected}.{expected_schema}.wiki_pages_chunks_index"
    )


def test_raw_pages_writer_uses_configured_catalog(monkeypatch):
    monkeypatch.setitem(config.wiki_config, "CATALOG", "wiki_test")
    monkeypatch.setitem(config.wiki_config, "SCHEMA", "research_test")
    monkeypatch.setattr(data_loader, "load_wikipedia_pages_async", AsyncMock(return_value=[]))
    spark = MagicMock()
    spark.catalog.tableExists.return_value = False

    asyncio.run(data_loader.sync_wikipedia_pages(spark))

    table_name = f"`{config.wiki_config['CATALOG']}`.`{config.wiki_config['SCHEMA']}`.`wiki_pages`"
    spark.catalog.tableExists.assert_called_once_with(table_name)
    writer = spark.createDataFrame.return_value.write.format.return_value.mode.return_value
    writer.option.return_value.saveAsTable.assert_called_once_with(
        table_name
    )


@pytest.mark.parametrize("override", [None, "explicit_catalog"])
@pytest.mark.parametrize("schema_override", [None, "explicit_schema"])
def test_chunk_sync_uses_same_source_and_destination_catalog(monkeypatch, override, schema_override):
    monkeypatch.setitem(config.wiki_config, "CATALOG", "wiki_test")
    monkeypatch.setitem(config.wiki_config, "SCHEMA", "research_test")
    load = MagicMock()
    load.return_value.filter.return_value.limit.return_value.count.return_value = 0
    monkeypatch.setattr(indexing, "load_wikipedia_chunks", load)
    monkeypatch.setattr(indexing, "col", MagicMock())
    spark = MagicMock()
    spark.catalog.tableExists.return_value = False

    indexing.sync_wikipedia_chunks(spark, catalog=override, schema=schema_override)

    expected = override or config.wiki_config["CATALOG"]
    expected_schema = schema_override or config.wiki_config["SCHEMA"]
    load.assert_called_once_with(spark, table_name=f"{expected}.{expected_schema}.wiki_pages")
    spark.catalog.tableExists.assert_called_once_with(
        f'`{expected}`.`{expected_schema}`.`wiki_pages_chunks`'
    )


def test_chunk_loader_reads_configured_catalog(monkeypatch):
    monkeypatch.setitem(config.wiki_config, "CATALOG", "wiki_test")
    monkeypatch.setitem(config.wiki_config, "SCHEMA", "research_test")
    monkeypatch.setattr(indexing, "recursive_split_udf", MagicMock())
    monkeypatch.setattr(indexing, "posexplode", MagicMock())
    monkeypatch.setattr(indexing, "col", MagicMock())
    monkeypatch.setattr(indexing, "concat_ws", MagicMock())
    spark = MagicMock()

    indexing.load_wikipedia_chunks(spark)

    spark.table.assert_called_once_with(
        f"{config.wiki_config['CATALOG']}.{config.wiki_config['SCHEMA']}.wiki_pages"
    )
