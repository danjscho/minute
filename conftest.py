import asyncio
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):  # noqa: ARG001
    """Patch queue service instantiation before any module-level imports trigger it.

    Routes like chat.py call get_queue_service() at import time. Without this patch,
    importing the FastAPI app during test collection attempts a live SQS connection.
    """
    _queue_patcher = patch(
        "common.services.queue_services.get_queue_service",
        return_value=MagicMock(),
    )
    _queue_patcher.start()


@pytest.fixture(scope="session", autouse=True)
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
