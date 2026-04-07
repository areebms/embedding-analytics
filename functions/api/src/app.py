import json
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache as fastapi_cache_decorator
from fastapi_cache.backends.redis import RedisBackend
from mangum import Mangum
from pydantic import BaseModel
from redis import asyncio as aioredis

from shared.aws import get_pipeline_table, TermTable, get_session
from main import extract_vectors, get_confidence_intervals, normalize_vector_bytes


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL")
REDIS_PREFIX = os.environ["REDIS_PREFIX"]
PRODUCTION_DOMAIN = os.environ.get("PRODUCTION_DOMAIN")


def cache(**kwargs):
    if REDIS_URL:
        return fastapi_cache_decorator(**kwargs)
    return lambda f: f


@asynccontextmanager
async def lifespan(app: FastAPI):
    if REDIS_URL:
        redis = aioredis.from_url(REDIS_URL)
        # Startup connectivity check
        await redis.ping()
        # Initialize global cache to use Redis
        FastAPICache.init(RedisBackend(redis), prefix=REDIS_PREFIX)
        yield
        await redis.close()
    else:
        yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()

    # Log similarity queries before the cache layer intercepts them,
    # so we see every request regardless of cache hit/miss.
    if request.method == "POST" and request.url.path.startswith("/similarity/"):
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
            book_id = request.url.path.removeprefix("/similarity/")
            logger.info(
                "similarity book_id=%s primary_term=%r secondary_term=%r",
                book_id,
                body.get("primary_term"),
                body.get("secondary_term"),
            )
        except Exception:
            pass
        # Reconstruct the request so the endpoint can still read the body.
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request = Request(request.scope, receive)

    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s status=%d duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", PRODUCTION_DOMAIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class Query(BaseModel):
    primary_term: str
    secondary_term: str | None = None


@app.post("/similarity/{book_id}")
@cache(expire=None)
def similarity(book_id: str, query: Query):
    platform_data = f"gutenberg-{book_id}"
    table = TermTable(get_session())

    primary_term_data = table.get_entry(query.primary_term, platform_data, ["vectors"])
    primary_raw = extract_vectors(primary_term_data["vectors"])

    if query.secondary_term:
        secondary_term_data = table.get_entry(
            query.secondary_term, platform_data, ["vectors"]
        )
        secondary_raw = extract_vectors(secondary_term_data["vectors"])
        combined = np.mean(np.stack([primary_raw, secondary_raw]), axis=0)
        term_vectors = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    else:
        term_vectors = primary_raw / np.linalg.norm(primary_raw, axis=1, keepdims=True)

    table_data = []
    for item_data in table.get_entries(
        platform_data, fields=["term", "count_", "tags", "vectors"]
    ):
        if item_data["tags"] == {"R"}:
            continue

        cosine_similarity, ci_half = get_confidence_intervals(
            term_vectors, normalize_vector_bytes(item_data["vectors"])
        )

        table_data.append(
            {
                "term": item_data["term"],
                "pos": item_data["tags"],
                "count": int(item_data["count_"]),
                "similarity": cosine_similarity,
                "similarity_ci": [
                    cosine_similarity - ci_half,
                    cosine_similarity + ci_half,
                ],
            }
        )
    return table_data


@app.get("/books")
@cache(expire=None)
def books():
    return [
        {
            "id": int(item["platform_data"].split("-")[-1]),
            "label": f"{item['author'].split(',')[0]} ({item['published_year']})",
            "author": item["author"],
            "title": item["title"],
            "published_year": item["published_year"],
        }
        for item in get_pipeline_table().get_all_entries(
            [
                "platform_data",
                "author",
                "published_year",
                "title",
                "s3_prefix_models",
            ]
        )
        if "s3_prefix_models" in item
    ]


handler = Mangum(app, lifespan="on")
