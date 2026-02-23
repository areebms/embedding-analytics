import json
from shared.aws import load_text_from_s3

def publish(table, session, index):
    item = table.get(index, expression="s3_metadata_key")
    s3_metadata_key = item.get("s3_metadata_key")
    if not s3_metadata_key:
        print(f"{index} has not been scraped.")
        exit()

    metadata = json.loads(load_text_from_s3(session, s3_metadata_key))

    table.update_entry(index, "author", ";".join(metadata["author"]))
    table.update_entry(index, "title", metadata["title"][0])
