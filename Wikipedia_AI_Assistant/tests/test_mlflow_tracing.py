import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from langchain_core.documents import Document

from src import wiki_agent as agent


class TestRetrievalSmokeTest(unittest.TestCase):
    def test_prints_documents_and_enables_autolog_without_tracking_configuration(self):
        docs = [Document(page_content="Test context", metadata={"chunk_id": "1_0"})]
        output = io.StringIO()
        with (
            patch.object(agent, "mlflow") as mlflow,
            patch.object(agent, "retrieve_wikipedia_documents") as tool,
            redirect_stdout(output),
        ):
            mlflow.get_last_active_trace_id.return_value = "tr-test"
            tool.invoke.return_value = docs
            result = agent.test_retrieval_tool()
            mlflow.langchain.autolog.assert_called_once_with()
            mlflow.set_tracking_uri.assert_not_called()
            mlflow.set_experiment.assert_not_called()
            tool.invoke.assert_called_once_with(
                {"input": "What causes stick-slip vibration during drilling?"}
            )
        self.assertIs(result, docs)
        self.assertIn("Test context", output.getvalue())
        self.assertIn("tr-test", output.getvalue())

    def test_prints_no_results_message(self):
        output = io.StringIO()
        with (
            patch.object(agent, "mlflow"),
            patch.object(agent, "retrieve_wikipedia_documents") as tool,
            redirect_stdout(output),
        ):
            tool.invoke.return_value = agent.NO_RELEVANT_DOCUMENTS
            result = agent.test_retrieval_tool("An unrelated question")
        self.assertEqual(result, agent.NO_RELEVANT_DOCUMENTS)
        self.assertIn(agent.NO_RELEVANT_DOCUMENTS, output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
