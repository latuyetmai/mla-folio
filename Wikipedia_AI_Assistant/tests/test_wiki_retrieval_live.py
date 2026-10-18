"""Opt-in live test: WIKI_LIVE_TEST=1 python -m unittest discover -s tests -p test_wiki_retrieval_live.py -v."""

import io
import json
import logging
import os
import unittest
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

from src.wiki_agent import (
    NO_RELEVANT_DOCUMENTS,
    retrieve_wikipedia_documents,
)


@unittest.skipUnless(
    os.getenv("WIKI_LIVE_TEST") == "1", "Set WIKI_LIVE_TEST=1 to query Databricks"
)
class TestLiveWikipediaRetrieval(unittest.TestCase):
    def test_stick_slip_question(self):
        logger = logging.getLogger("src.wiki_agent")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        diagnostics = io.StringIO()
        capture = logging.StreamHandler(diagnostics)
        logger.addHandler(capture)
        self.addCleanup(logger.removeHandler, capture)

        mlflow.langchain.autolog()
        question = "What causes stick-slip vibration during drilling?"
        result = retrieve_wikipedia_documents.invoke({"input": question})
        trace_id = mlflow.get_last_active_trace_id()
        print(f"MLflow trace ID: {trace_id}", flush=True)
        if isinstance(result, str):
            self.assertEqual(result, NO_RELEVANT_DOCUMENTS)
            documents = []
        else:
            self.assertTrue(all(isinstance(doc, Document) for doc in result))
            documents = [doc.model_dump(mode="json") for doc in result]
        output = {
            "question": question,
            "document_count": len(documents),
            "documents": documents,
            "message": result if isinstance(result, str) else None,
            "diagnostics": diagnostics.getvalue().splitlines(),
            "trace_id": trace_id,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        if output_path := os.getenv("WIKI_RETRIEVAL_OUTPUT"):
            Path(output_path).write_text(
                json.dumps(output, indent=2, ensure_ascii=False)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
