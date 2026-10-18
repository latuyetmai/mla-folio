import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from uuid import uuid4

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from src import wiki_agent as agent


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class LoopingModel(ToolCallingModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "retrieve_wikipedia_documents",
                                "args": {"input": "horizontal drilling"},
                                "id": uuid4().hex,
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


class FailingModel(ToolCallingModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("Primary model unavailable")


class TestReActAgent(unittest.TestCase):
    def build_with_model(self, model, fallback=None):
        self.models = []

        def get_model(endpoint, **kwargs):
            if endpoint == agent.config.wiki_config["LLM_ENDPOINT_NAME"]:
                result = model
            else:
                result = fallback or ToolCallingModel(
                    responses=[AIMessage(content="Summary or fallback")]
                )
            self.models.append((endpoint, result))
            return result

        with (
            patch.object(agent, "_workspace_client"),
            patch.object(agent, "ChatDatabricks", side_effect=get_model),
        ):
            return agent.create_wiki_agent(checkpointer=InMemorySaver())

    def test_exact_middleware_configuration_and_prompt(self):
        model = ToolCallingModel(responses=[AIMessage(content="Answer")])
        with patch.object(agent, "create_agent") as create:
            self.build_with_model(model)
        kwargs = create.call_args.kwargs
        summary, tool_limit, model_limit, fallback = kwargs["middleware"]
        self.assertIsInstance(summary, SummarizationMiddleware)
        self.assertEqual(summary.trigger, [("tokens", 150_000), ("messages", 50)])
        self.assertEqual(summary.keep, ("tokens", 5_000))
        self.assertEqual(summary.trim_tokens_to_summarize, 25_000)
        self.assertEqual(
            self.models[1][0],
            agent.config.wiki_config["SUMMARIZATION_LLM_ENDPOINT_NAME"],
        )
        self.assertIsInstance(tool_limit, ToolCallLimitMiddleware)
        self.assertEqual(
            (tool_limit.tool_name, tool_limit.run_limit, tool_limit.exit_behavior),
            ("retrieve_wikipedia_documents", 5, "continue"),
        )
        self.assertIsInstance(model_limit, ModelCallLimitMiddleware)
        self.assertEqual(
            (
                model_limit.thread_limit,
                model_limit.run_limit,
                model_limit.exit_behavior,
            ),
            (50, 10, "end"),
        )
        self.assertIsInstance(fallback, ModelFallbackMiddleware)
        self.assertEqual(len(fallback.models), 2)
        self.assertEqual(
            [endpoint for endpoint, _ in self.models[2:]],
            [
                agent.config.wiki_config["LLM_FALLBACK_ENDPOINT_NAME"],
                agent.config.wiki_config["LLM_FALLBACK_ENDPOINT_NAME2"],
            ],
        )
        self.assertEqual(kwargs["tools"], [agent.retrieve_wikipedia_documents])
        self.assertIsInstance(kwargs["checkpointer"], InMemorySaver)
        self.assertIsInstance(kwargs["system_prompt"], SystemMessage)
        prompt_block = kwargs["system_prompt"].content[0]
        self.assertEqual(prompt_block["type"], "text")
        self.assertEqual(prompt_block["cache_control"], {"type": "ephemeral"})
        self.assertIn(
            "I was not able to retrieve a relevant document.", prompt_block["text"]
        )
        self.assertIn("Based on general knowledge,", prompt_block["text"])

    def test_real_graph_runs_retrieval_then_returns_answer(self):
        model = ToolCallingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_wikipedia_documents",
                            "args": {"input": "horizontal drilling"},
                            "id": "retrieve-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Horizontal drilling follows a reservoir laterally."),
            ]
        )
        graph = self.build_with_model(model)
        doc = Document(
            page_content="A horizontal well follows the reservoir.",
            metadata={
                "title": "Directional drilling",
                "fullurl": "https://en.wikipedia.org/wiki/Directional_drilling",
            },
        )
        with patch.object(agent, "retrieval_chain") as retrieval:
            retrieval.invoke.return_value = [doc]
            events = list(
                graph.stream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Tell me about horizontal drilling",
                            }
                        ]
                    },
                    config={"configurable": {"thread_id": "test-loop"}},
                    stream_mode="updates",
                )
            )
            retrieval.invoke.assert_called_once_with("horizontal drilling")
        self.assertTrue(any("tools" in event for event in events))
        state = graph.get_state({"configurable": {"thread_id": "test-loop"}}).values
        tool_message = next(
            message for message in state["messages"] if isinstance(message, ToolMessage)
        )
        self.assertIn("follows the reservoir", tool_message.content)
        self.assertIn("laterally", state["messages"][-1].content)

    def test_run_limits_stop_repeated_model_calls_and_block_sixth_tool_call(self):
        graph = self.build_with_model(
            LoopingModel(responses=[AIMessage(content="unused")])
        )
        with patch.object(agent, "retrieval_chain") as retrieval:
            retrieval.invoke.return_value = agent.NO_RELEVANT_DOCUMENTS
            result = graph.invoke(
                {"messages": [{"role": "user", "content": "horizontal drilling"}]},
                config={
                    "configurable": {"thread_id": "limit-test"},
                    "recursion_limit": 100,
                },
            )
            self.assertEqual(retrieval.invoke.call_count, 5)
        model_tool_requests = [
            message
            for message in result["messages"]
            if isinstance(message, AIMessage) and message.tool_calls
        ]
        self.assertEqual(len(model_tool_requests), 10)
        self.assertIn("limit", result["messages"][-1].content.lower())

    def test_model_fallback_recovers_primary_failure(self):
        graph = self.build_with_model(
            FailingModel(responses=[AIMessage(content="unused")]),
            fallback=ToolCallingModel(responses=[AIMessage(content="Fallback answer")]),
        )
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            config={"configurable": {"thread_id": "fallback-test"}},
        )
        self.assertEqual(result["messages"][-1].content, "Fallback answer")

    def test_streaming_smoke_test_uses_updates_mode(self):
        output = io.StringIO()
        with (
            patch.object(agent, "__getattr__") as get_agent,
            patch.object(agent, "mlflow"),
            redirect_stdout(output),
        ):
            get_agent.return_value.stream.return_value = iter(
                [{"model": {"messages": [AIMessage(content="Answer")]}}]
            )
            updates = agent.test_wiki_agent(thread_id="smoke-test")
        get_agent.assert_called_once_with("wiki_agent")
        call = get_agent.return_value.stream.call_args
        self.assertEqual(call.kwargs["stream_mode"], "updates")
        self.assertEqual(
            call.args[0]["messages"][0]["content"], "Tell me about horizontal drilling"
        )
        self.assertEqual(
            call.kwargs["config"]["configurable"]["thread_id"], "smoke-test"
        )
        self.assertEqual(len(updates), 1)
        self.assertIn("Answer", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
