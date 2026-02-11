import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from main import KeyedVectorGroup
from shared.aws import PipelineTable, get_session

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    os.environ.get("PRODUCTION_DOMAIN"),
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,  # Allows cookies to be sent cross-origin
    allow_methods=["GET"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/similarity/{book_id}/{primary_term}")
def similarity(primary_term: str, book_id):
    keyed_vector_group = KeyedVectorGroup(index=f"gutenberg-{book_id}")
    keyed_vector_group.fetch_precalculated_data()
    table_data = []
    for term, stability_data in keyed_vector_group.term_stability_data.items():
        table_data.append(
            {
                "term": term,
                "count": stability_data["count"],
                "coherence": 1 - float(stability_data["semantic"]),
                "similarity": float(
                    keyed_vector_group.centroid.similarity(term, primary_term)
                ),
            }
        )
    return table_data


@app.get("/books")
def books():
    return [
        {
            "id": int(item["platform_data"].split("-")[-1]),
            "label": f"{item['author'].split(',')[0]} ({item['published_year']})",
            "author": item["author"],
            "title": item["title"],
        }
        for item in PipelineTable(get_session()).get_all(
            ["platform_data", "author", "published_year", "title"]
        )
    ]

handler = Mangum(app)