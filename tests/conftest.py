from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
import pytest_asyncio

from runtime.models import RuntimeConfig
from runtime.store import SQLiteJobStore


@pytest_asyncio.fixture
async def store_factory(tmp_path: Path) -> AsyncIterator[Callable[..., SQLiteJobStore]]:
    stores: list[SQLiteJobStore] = []

    def make_store(
        *,
        filename: str = "jobs.sqlite3",
        random_source: Callable[[], float] = lambda: 1.0,
        **config_overrides,
    ) -> SQLiteJobStore:
        config = RuntimeConfig(db_path=str(tmp_path / filename), **config_overrides)
        store = SQLiteJobStore(config, random_source=random_source)
        stores.append(store)
        return store

    yield make_store

    for store in reversed(stores):
        await store.close()


@pytest_asyncio.fixture
async def store(store_factory) -> SQLiteJobStore:
    instance = store_factory()
    await instance.start()
    return instance


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "jobs.sqlite3"
