import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from mangum import Mangum
from redis import asyncio as aioredis

from main import KeyedVectorGroup
from shared.aws import PipelineTable, get_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = aioredis.from_url(os.environ["REDIS_URL"])
    # Startup connectivity check
    await redis.ping()

    # Initialize global cache to use Redis
    FastAPICache.init(RedisBackend(redis), prefix=os.environ["REDIS_PREFIX"])
    yield
    await redis.close()


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    os.environ.get("PRODUCTION_DOMAIN"),
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,  # Allows cookies to be sent cross-origin
    allow_methods=["GET", "OPTIONS"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/similarity/{book_id}/{primary_term}")
@cache(expire=None)
def similarity(primary_term: str, book_id):
    keyed_vector_group = KeyedVectorGroup(index=f"gutenberg-{book_id}")
    keyed_vector_group.fetch_precalculated_data()
    table_data = []
    
    for term in list(keyed_vector_group.centroid.key_to_index):
        table_data.append(
            {
                "term": term,
                "count": keyed_vector_group.centroid[term]["count"],
                "coherence": 1 - float(keyed_vector_group.centroid[term]["count"]),
                "similarity": float(
                    keyed_vector_group.centroid.similarity(term, primary_term)
                ),
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
        }
        for item in PipelineTable(get_session()).get_all(
            [
                "platform_data",
                "author",
                "published_year",
                "title",
                "s3_prefix_models",
            ]
        )
      if 's3_prefix_models' in item
    ]


handler = Mangum(app, lifespan="on")
