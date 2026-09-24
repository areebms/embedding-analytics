from collections import defaultdict
from time import sleep

from bs4 import BeautifulSoup
from requests import get

MAX_BOOK_IDS_PER_PAGE = 25
MAX_BOOKS_PER_SUBJECT = 100
BASE_URL = "https://gutenberg.org"


def get_book_ids(subject_id):
    """A subject's book ids, most downloaded first, capped at MAX_BOOKS_PER_SUBJECT.

    The length check on the href matters: an `ebooks` link with no path of its own
    (a nav link, say) has nothing at index 2 to test.
    """
    url = f"{BASE_URL}/ebooks/subject/{subject_id}/?sort_order=downloads"

    book_ids = []
    start_index = 1

    while len(book_ids) < MAX_BOOKS_PER_SUBJECT:
        sleep(1)
        response = get(f"{url}&start_index={start_index}")
        response.raise_for_status()
        start_index += MAX_BOOK_IDS_PER_PAGE

        new_book_ids = []
        for link in BeautifulSoup(response.text, "html.parser").find_all("a"):
            href = link.get("href", "")
            if "ebooks" not in href:
                continue
            path = href.split("/")
            if len(path) <= 2 or not path[2].isdigit():
                continue
            if path[2] not in book_ids and path[2] not in new_book_ids:
                new_book_ids.append(path[2])

        if not new_book_ids:
            break

        book_ids.extend(new_book_ids)

    return book_ids[:MAX_BOOKS_PER_SUBJECT]


def get_metadata(gutenberg_id):
    response = get(f"{BASE_URL}/ebooks/{gutenberg_id}/")
    response.raise_for_status()

    metadata_table = BeautifulSoup(response.content, "html.parser").find(
        "table", class_="bibrec"
    )
    if metadata_table is None:
        raise ValueError(f"No metadata table for gutenberg id {gutenberg_id}")

    metadata = defaultdict(list)
    for table_row in metadata_table.find_all("tr"):
        table_header = table_row.find("th")
        table_data = table_row.find("td")
        if table_header is None or table_data is None:
            continue

        a = table_data.find("a")
        href = a["href"] if a else None
        key = (
            table_header.text.lower()
            .replace(" ", "-")
            .replace(".", "")
            .replace("-", "_")
        )
        for line in table_data.get_text(separator="\n").split("\n"):
            if line:
                metadata[key].append(line)
        if href:
            metadata[key + "_link"].append(href)

    return dict(metadata)


def get_html(gutenberg_id):
    response = get(f"{BASE_URL}/cache/epub/{gutenberg_id}/pg{gutenberg_id}-images.html")
    response.raise_for_status()
    return response.text
