import os

from fastapi_cache.decorator import cache as fastapi_cache_decorator

REDIS_URL = os.environ.get("REDIS_URL")

def cache(**kwargs):
    if REDIS_URL:
        return fastapi_cache_decorator(**kwargs)
    return lambda f: f
