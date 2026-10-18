from unittest.mock import MagicMock

import pytest

from src import indexing


@pytest.fixture
def settings(monkeypatch):
    values = {
        "CATALOG": indexing.config.wiki_config["CATALOG"],
        "SCHEMA": indexing.config.wiki_config["SCHEMA"],
        "VECTORSEARCH_ENDPOINT": "wiki-search",
        "EMBEDDING_MODEL": "databricks-bge-large-en",
    }
    monkeypatch.setattr(indexing.config, "wiki_config", values)
    return values


def test_create_index_with_all_source_columns(settings):
    source_table_name = f"{settings['CATALOG']}.{settings['SCHEMA']}.wiki_pages_chunks"
    index_name = f"{source_table_name}_index"
    spark = MagicMock()
    columns = ["chunk_id", "text_chunks", "pageid", "title", "updated_at"]
    spark.table.return_value.columns = columns
    client = MagicMock()
    client.index_exists.return_value = False

    result = indexing.create_vector_index(spark, client=client)

    client.index_exists.assert_called_once_with(
        endpoint_name="wiki-search",
        index_name=index_name,
    )
    spark.table.assert_called_once_with(source_table_name)
    client.create_delta_sync_index.assert_called_once_with(
        endpoint_name="wiki-search",
        source_table_name=source_table_name,
        index_name=index_name,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="text_chunks",
        embedding_model_endpoint_name="databricks-bge-large-en",
        columns_to_sync=columns,
    )
    client.get_index.assert_not_called()
    assert result is client.create_delta_sync_index.return_value


def test_existing_index_syncs_without_recreation(settings):
    spark = MagicMock()
    client = MagicMock()
    client.index_exists.return_value = True

    result = indexing.create_vector_index(spark, "catalog.schema.chunks", client=client)

    client.get_index.assert_called_once_with(
        endpoint_name="wiki-search", index_name="catalog.schema.chunks_index"
    )
    result.sync.assert_called_once_with()
    client.create_delta_sync_index.assert_not_called()
    spark.table.assert_not_called()


@pytest.mark.parametrize("key", ["VECTORSEARCH_ENDPOINT", "EMBEDDING_MODEL"])
def test_missing_settings_fail_before_api_calls(settings, key):
    settings.pop(key)
    client = MagicMock()
    with pytest.raises(ValueError, match=key):
        indexing.create_vector_index(MagicMock(), client=client)
    client.index_exists.assert_not_called()


def test_missing_source_columns_do_not_create_index(settings):
    spark = MagicMock()
    spark.table.return_value.columns = ["pageid", "title"]
    client = MagicMock()
    client.index_exists.return_value = False
    with pytest.raises(ValueError, match="chunk_id and text_chunks"):
        indexing.create_vector_index(spark, client=client)
    client.create_delta_sync_index.assert_not_called()
