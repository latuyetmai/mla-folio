"""Offline tests; all model and Vector Search calls are mocked."""

import json
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src import wiki_agent as agent

QUESTION = "What causes stick-slip vibration during drilling?"


def sample_docs(count=3):
    return [
        Document(
            page_content=f"Synthetic test context {i}: torsional drill-string vibration.",
            metadata={
                "chunk_id": f"1_{i}",
                "title": "Test page",
                "topic": "stick-slip drilling",
                "fullurl": "https://example.test/wiki",
                "summary": "Synthetic summary",
            },
        )
        for i in range(count)
    ]


class TestWikipediaRetrieval(unittest.TestCase):
    def test_prefilter_returns_validated_topic_dictionary_and_logs(self):
        classifier = MagicMock()
        classifier.invoke.return_value = agent.TopicSelection(
            topic=["stick-slip drilling"]
        )
        with (
            patch.object(agent, "_classification_chain", return_value=classifier),
            self.assertLogs(agent.logger, level="INFO") as logs,
        ):
            result = agent.pre_filter_topics(QUESTION)
        self.assertEqual(result, {"topic": ["stick-slip drilling"]})
        prompt = classifier.invoke.call_args.args[0][0].content
        self.assertTrue(all(topic in prompt for topic in agent.wiki_topics))
        self.assertIn("filter=", " ".join(logs.output))

    def test_unknown_topics_are_removed_and_valid_topics_are_deduplicated(self):
        result = agent.TopicSelection(
            topic=["horizontal drilling", "lateral drilling", "stick-slip drilling", "horizontal drilling"]
        )
        self.assertEqual(result.topic, ["horizontal drilling", "stick-slip drilling"])
        self.assertEqual(agent.TopicSelection(topic=["an invented topic"]).topic, [])

    def test_prefilter_parses_with_topic_selection_and_removes_unknown_topics(self):
        with (
            patch.object(agent, "ChatDatabricks") as model,
            patch.object(agent, "_workspace_client"),
        ):
            model.return_value.with_structured_output.return_value.invoke.return_value = {
                "topic": ["horizontal drilling", "lateral drilling"]
            }
            self.assertEqual(agent.pre_filter_topics(QUESTION), {"topic": ["horizontal drilling"]})
            model.return_value.with_structured_output.assert_called_once_with(
                agent.TopicSelection, method="json_mode"
            )

    def test_classification_failure_uses_fallback_model(self):
        with (
            patch.dict(
                agent.config.wiki_config,
                {
                    "CLASSIFICATION_LLM_ENDPOINT_NAME": "classifier",
                    "LLM_FALLBACK_ENDPOINT_NAME": "fallback",
                },
            ),
            patch.object(agent, "ChatDatabricks") as model,
            patch.object(agent, "_workspace_client"),
        ):
            model.return_value.with_structured_output.return_value.invoke.side_effect = [
                RuntimeError("primary unavailable"),
                {"topic": ["stick-slip drilling"]},
            ]
            result = agent.pre_filter_topics(QUESTION)
            self.assertEqual(result["topic"], ["stick-slip drilling"])
            self.assertEqual(
                [call.kwargs["endpoint"] for call in model.call_args_list],
                ["classifier", "fallback"],
            )

    def test_invalid_structured_response_also_uses_fallback(self):
        with (
            patch.object(agent, "ChatDatabricks") as model,
            patch.object(agent, "_workspace_client"),
        ):
            model.return_value.with_structured_output.return_value.invoke.side_effect = [
                {"topic": "not a list"},
                {"topic": []},
            ]
            self.assertEqual(agent.pre_filter_topics(QUESTION), {"topic": []})
            self.assertEqual(model.call_count, 2)

    def test_service_principal_takes_precedence_over_token(self):
        with patch.dict(
            os.environ,
            {
                "DATABRICKS_HOST": "https://workspace.test/",
                "DATABRICKS_CLIENT_ID": "client",
                "DATABRICKS_CLIENT_SECRET": "secret",
                "DATABRICKS_TOKEN": "token",
            },
            clear=True,
        ):
            args = agent._client_args()
        self.assertEqual(
            args,
            {
                "workspace_url": "https://workspace.test",
                "service_principal_client_id": "client",
                "service_principal_client_secret": "secret",
            },
        )

    def test_token_auth_and_hostname_normalization(self):
        with patch.dict(
            os.environ,
            {"DATABRICKS_HOSTNAME": "workspace.test", "DATABRICKS_TOKEN": "token"},
            clear=True,
        ):
            self.assertEqual(
                agent._client_args(),
                {
                    "workspace_url": "https://workspace.test",
                    "personal_access_token": "token",
                },
            )

    def test_missing_service_principal_secret_is_not_silently_ignored(self):
        with (
            patch.dict(
                os.environ,
                {
                    "DATABRICKS_HOST": "https://workspace.test",
                    "DATABRICKS_CLIENT_ID": "client",
                },
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "DATABRICKS_CLIENT_SECRET"),
        ):
            agent._client_args()

    def test_hybrid_search_metadata_filter_reranker_and_top_ten(self):
        docs = sample_docs(20)
        topic_filter = {"topic": ["stick-slip drilling"]}
        with (
            patch.object(agent, "DatabricksVectorSearch") as store,
            patch.object(
                agent, "_client_args", return_value={"personal_access_token": "test"}
            ),
        ):
            store.return_value.similarity_search.return_value = docs
            result = agent.retrieve_relevant_docs(QUESTION, topic_filter)
            self.assertEqual(result, docs[:10])
            self.assertEqual(
                store.call_args.kwargs["index_name"],
                agent.config.wiki_config["VECTORSEARCH_WIKI_INDEX"],
            )
            self.assertEqual(store.call_args.kwargs["columns"], agent.METADATA_COLUMNS)
            # Managed Delta Sync indexes infer text_chunks from index metadata.
            self.assertNotIn("text_column", store.call_args.kwargs)
            kwargs = store.return_value.similarity_search.call_args.kwargs
            self.assertEqual(
                (kwargs["k"], kwargs["query_type"], kwargs["filter"]),
                (20, "HYBRID", topic_filter),
            )
            self.assertEqual(kwargs["reranker"].columns_to_rerank, ["summary"])

    def test_no_topics_skips_vector_search(self):
        with patch.object(agent, "DatabricksVectorSearch") as store:
            self.assertEqual(agent.retrieve_relevant_docs(QUESTION, {"topic": []}), [])
            store.assert_not_called()

    def test_postfilter_calls_one_document_per_call_concurrently(self):
        docs = sample_docs()
        barrier = threading.Barrier(3)

        def classify(messages):
            data = json.loads(messages[1].content)
            self.assertEqual(data["question"], QUESTION)
            barrier.wait(timeout=3)
            return agent.DocumentRelevance(
                relevant=data["content"] != docs[1].page_content,
                reason="Test classification",
            )

        with patch.object(
            agent, "_classification_chain", return_value=RunnableLambda(classify)
        ):
            result = agent.post_filter_docs(QUESTION, docs, max_concurrency=3)
        self.assertEqual(result, [docs[0], docs[2]])
        self.assertIs(result[0], docs[0])

    def test_all_irrelevant_and_empty_inputs_return_message(self):
        chain = RunnableLambda(
            lambda _: agent.DocumentRelevance(relevant=False, reason="Unrelated")
        )
        with patch.object(agent, "_classification_chain", return_value=chain):
            self.assertEqual(
                agent.post_filter_docs(QUESTION, sample_docs()),
                agent.NO_RELEVANT_DOCUMENTS,
            )
            self.assertEqual(
                agent.post_filter_docs(QUESTION, []), agent.NO_RELEVANT_DOCUMENTS
            )

    def test_tool_calls_all_three_steps_in_order_with_documents(self):
        docs = sample_docs()
        events = []
        topic_filter = {"topic": ["stick-slip drilling"]}

        def pre(question):
            events.append("pre")
            self.assertEqual(question, QUESTION)
            return topic_filter

        def retrieve(question, filters, **kwargs):
            events.append("retrieve")
            self.assertEqual((question, filters), (QUESTION, topic_filter))
            return docs

        def post(question, documents, **kwargs):
            events.append("post")
            self.assertEqual((question, documents), (QUESTION, docs))
            return docs[:1]

        with (
            patch.object(agent, "pre_filter_topics", side_effect=pre),
            patch.object(agent, "retrieve_relevant_docs", side_effect=retrieve),
            patch.object(agent, "post_filter_docs", side_effect=post),
        ):
            result = agent.retrieve_wikipedia_documents.invoke({"input": QUESTION})
        self.assertEqual(events, ["pre", "retrieve", "post"])
        self.assertEqual(result, docs[:1])
        self.assertIsInstance(result[0], Document)


if __name__ == "__main__":
    unittest.main(verbosity=2)
