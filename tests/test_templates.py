import pytest
from unittest.mock import MagicMock

from backend.api.dependencies.get_current_user import get_current_user
from backend.main import app
from common.database.postgres_models import User
from common.services.template_manager import TemplateManager
from tests.utils import get_test_client


def _mock_user():
    user = MagicMock(spec=User)
    user.id = "test-user-id"
    user.email = "test@test.co.uk"
    return user


async def _override_get_current_user():
    return _mock_user()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("expected_status_code", [200])
async def test_get_templates_success(expected_status_code):
    app.dependency_overrides[get_current_user] = _override_get_current_user
    try:
        async with get_test_client() as ac:
            response = await ac.get("/templates")
            assert response.status_code == expected_status_code
            assert len(response.json()) == len(TemplateManager.templates)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
