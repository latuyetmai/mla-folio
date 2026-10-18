"""Lakebase checkpoint storage with sync and async LangGraph interfaces."""

import asyncio
import atexit
import os

from databricks_langchain import CheckpointSaver


class LakebaseCheckpointSaver(CheckpointSaver):
    """Use the SDK's rotating-credential pool for both sync and async callers.

    CheckpointSaver implements synchronous Postgres operations. Async servers
    can use the same graph through these adapters without blocking their loop.
    """

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        iterator = self.list(config, filter=filter, before=before, limit=limit)
        sentinel = object()
        try:
            while (
                item := await asyncio.to_thread(next, iterator, sentinel)
            ) is not sentinel:
                yield item
        finally:
            await asyncio.to_thread(iterator.close)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(
            self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)


def create_lakebase_checkpointer(workspace_client) -> LakebaseCheckpointSaver:
    """Open a process-lived pool and initialize the LangGraph checkpoint tables."""
    instance_name = os.getenv("LAKEBASE_INSTANCE_NAME")
    host = os.getenv("LAKEBASE_HOST")
    if not instance_name or not host:
        raise ValueError("Set both LAKEBASE_INSTANCE_NAME and LAKEBASE_HOST")

    # The SDK resolves its connection host from the instance API; it does not
    # accept a host parameter. Verify the supplied host before opening a pool.
    instance = workspace_client.database.get_database_instance(name=instance_name)
    resolved_host = instance.read_write_dns or instance.read_only_dns
    if (
        not resolved_host
        or host.strip().rstrip(".").lower() != resolved_host.rstrip(".").lower()
    ):
        raise ValueError(
            "LAKEBASE_HOST does not match the configured Lakebase instance"
        )

    saver = LakebaseCheckpointSaver(
        instance_name=instance_name,
        workspace_client=workspace_client,
        min_size=1,
        max_size=10,
        timeout=30,
    )
    try:
        saver.setup()
    except Exception:
        saver.__exit__(None, None, None)
        raise
    # Do not exit a 'with CheckpointSaver(...)' block before the exported graph
    # is used: its connection pool must remain open throughout server lifetime.
    atexit.register(saver.__exit__, None, None, None)
    return saver
