import os
import unittest
from unittest.mock import MagicMock, patch

from src import lakebase_checkpoint as lakebase
from src import wiki_agent as agent


class TestLakebaseInitialization(unittest.TestCase):
    def test_environment_host_validation_setup_and_process_lifetime(self):
        workspace = MagicMock()
        workspace.database.get_database_instance.return_value.read_write_dns = (
            "lakebase.test"
        )
        with (
            patch.dict(
                os.environ,
                {
                    "LAKEBASE_INSTANCE_NAME": "wiki-memory",
                    "LAKEBASE_HOST": "lakebase.test",
                },
            ),
            patch.object(lakebase, "LakebaseCheckpointSaver") as saver,
            patch.object(lakebase.atexit, "register") as register,
        ):
            result = lakebase.create_lakebase_checkpointer(workspace)
            workspace.database.get_database_instance.assert_called_once_with(
                name="wiki-memory"
            )
            saver.assert_called_once_with(
                instance_name="wiki-memory",
                workspace_client=workspace,
                min_size=1,
                max_size=10,
                timeout=30,
            )
            result.setup.assert_called_once_with()
            result.__exit__.assert_not_called()
            register.assert_called_once_with(result.__exit__, None, None, None)

    def test_missing_environment_fails_before_opening_pool(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(lakebase, "LakebaseCheckpointSaver") as saver,
        ):
            with self.assertRaisesRegex(ValueError, "LAKEBASE_INSTANCE_NAME"):
                lakebase.create_lakebase_checkpointer(MagicMock())
            saver.assert_not_called()

    def test_mismatched_host_is_rejected(self):
        workspace = MagicMock()
        workspace.database.get_database_instance.return_value.read_write_dns = (
            "another.test"
        )
        with (
            patch.dict(
                os.environ,
                {
                    "LAKEBASE_INSTANCE_NAME": "wiki-memory",
                    "LAKEBASE_HOST": "lakebase.test",
                },
            ),
            patch.object(lakebase, "LakebaseCheckpointSaver") as saver,
        ):
            with self.assertRaisesRegex(ValueError, "does not match"):
                lakebase.create_lakebase_checkpointer(workspace)
            saver.assert_not_called()

    def test_failed_setup_closes_the_pool(self):
        workspace = MagicMock()
        workspace.database.get_database_instance.return_value.read_write_dns = (
            "lakebase.test"
        )
        with (
            patch.dict(
                os.environ,
                {
                    "LAKEBASE_INSTANCE_NAME": "wiki-memory",
                    "LAKEBASE_HOST": "lakebase.test",
                },
            ),
            patch.object(lakebase, "LakebaseCheckpointSaver") as saver,
            patch.object(lakebase.atexit, "register") as register,
        ):
            saver.return_value.setup.side_effect = RuntimeError("setup failed")
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                lakebase.create_lakebase_checkpointer(workspace)
            saver.return_value.__exit__.assert_called_once_with(None, None, None)
            register.assert_not_called()

    def test_export_initializes_and_reuses_one_compiled_agent(self):
        with patch.dict(agent.__dict__):
            agent.__dict__.pop("wiki_agent", None)
            with patch.object(agent, "create_wiki_agent") as create:
                first = agent.__getattr__("wiki_agent")
                second = agent.__getattr__("wiki_agent")
                self.assertIs(first, create.return_value)
                self.assertIs(first, second)
                self.assertIs(agent.wiki_agent, first)
                create.assert_called_once_with()

    def test_default_factory_passes_lakebase_saver_to_create_agent(self):
        with (
            patch.object(agent, "_workspace_client") as workspace,
            patch.object(agent, "create_lakebase_checkpointer") as saver,
            patch.object(agent, "ChatDatabricks"),
            patch.object(agent, "create_agent") as create,
        ):
            agent.create_wiki_agent()
            saver.assert_called_once_with(workspace.return_value)
            self.assertIs(create.call_args.kwargs["checkpointer"], saver.return_value)


class TestAsyncCheckpointInterface(unittest.IsolatedAsyncioTestCase):
    async def test_async_get_put_writes_and_delete_delegate_to_sync_pool(self):
        saver = object.__new__(lakebase.LakebaseCheckpointSaver)
        config = {"configurable": {"thread_id": "test-thread"}}
        with (
            patch.object(saver, "get_tuple", return_value="checkpoint") as get,
            patch.object(saver, "put", return_value=config) as put,
            patch.object(saver, "put_writes") as writes,
            patch.object(saver, "delete_thread") as delete,
        ):
            self.assertEqual(await saver.aget_tuple(config), "checkpoint")
            self.assertEqual(await saver.aput(config, {}, {}, {}), config)
            await saver.aput_writes(config, [("messages", "hello")], "task", "path")
            await saver.adelete_thread("test-thread")
            get.assert_called_once_with(config)
            put.assert_called_once_with(config, {}, {}, {})
            writes.assert_called_once_with(
                config, [("messages", "hello")], "task", "path"
            )
            delete.assert_called_once_with("test-thread")

    async def test_async_list_yields_checkpoints(self):
        saver = object.__new__(lakebase.LakebaseCheckpointSaver)

        def checkpoints():
            yield "checkpoint-1"
            yield "checkpoint-2"

        with patch.object(saver, "list", return_value=checkpoints()) as listing:
            result = [item async for item in saver.alist(None, limit=2)]
        self.assertEqual(result, ["checkpoint-1", "checkpoint-2"])
        listing.assert_called_once_with(None, filter=None, before=None, limit=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
