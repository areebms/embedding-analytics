import json
import os

from shared.aws import PipelineTable, upload_object, get_session
from shared.commons import get_index
from retrieve import get_html, get_text, get_metadata


AWS_REGION = os.getenv("AWS_REGION")
AWS_PROFILE = os.getenv("AWS_PROFILE")


def scrape(index):

    session = get_session()
    table = PipelineTable(session)
    created = table.put(index)

    if not created:
        item = table.get(index, "platform_data, s3_text_key")
        if item and "s3_text_key" in item:
            print(f"{index} has already been scraped.")
            return

    html = get_html(index.split("-")[1])
    html_key = f"html/{index}.html"
    upload_object(
        session,
        html_key,
        html.encode("utf-8"),  # TODO: move encoding to function?
    )
    table.update_entry(index, "s3_html_key", html_key)

    metadata = get_metadata(index.split("-")[1])
    metadata_key = f"metadata/{index}.json"
    upload_object(
        session,
        metadata_key,
        json.dumps(metadata).encode("utf-8"),  # TODO: move encoding to function?
        content_type="application/json; charset=utf-8",
    )
    table.update_entry(index, "s3_metadata_key", metadata_key)

    text = get_text(html)
    text_key = f"text/{index}.txt"
    upload_object(session, text_key, text.encode("utf-8"))
    table.update_entry(index, "s3_text_key", text_key)
    print(f"{index} scraped.")


if __name__ == "__main__":
    scrape(get_index())
