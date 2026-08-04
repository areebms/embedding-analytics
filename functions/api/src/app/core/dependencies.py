import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache as fastapi_cache_decorator
from redis import asyncio as aioredis

from app.core.services import BooksMetadataCache
from shared.tables.pipeline import PipelineTable, get_pipeline_table

REDIS_URL = os.environ.get("REDIS_URL")


def cache(**kwargs) -> Callable[[Any], Any]:
    if REDIS_URL:
        return fastapi_cache_decorator(**kwargs)
    return lambda f: f


@asynccontextmanager
async def lifespan(app: FastAPI):
    if REDIS_URL:
        redis: aioredis.Redis = aioredis.from_url(REDIS_URL)
        await cast(Awaitable[Any], redis.ping())
        FastAPICache.init(RedisBackend(redis), prefix=os.environ["REDIS_PREFIX"])
        try:
            yield
        finally:
            await redis.aclose()
    else:
        yield


PipelineTableDep = Annotated[PipelineTable, Depends(get_pipeline_table)]

books_metadata_cache: BooksMetadataCache | None = None


def get_books_metadata_cache(table: PipelineTableDep) -> BooksMetadataCache:

    global books_metadata_cache
    if books_metadata_cache is None or books_metadata_cache.table is not table:
        books_metadata_cache = BooksMetadataCache(table)
    return books_metadata_cache


BooksMetadataCacheDep = Annotated[BooksMetadataCache, Depends(get_books_metadata_cache)]
