from collections import defaultdict
from time import sleep

from bs4 import BeautifulSoup
from requests import get

MAX_BOOK_IDS_PER_PAGE = 25
BASE_URL = "https://gutenberg.org"


def get_book_ids(subject_id):
    url = f"{BASE_URL}/ebooks/subject/{subject_id}/"

    page_book_ids = []
    num_book_ids = -1
    while num_book_ids == -1 or (
        len(page_book_ids) != num_book_ids
        and len(page_book_ids) % MAX_BOOK_IDS_PER_PAGE == 0
    ):
        sleep(1)
        num_book_ids = len(page_book_ids)
        book_list_page = get(url + f"?start_index={num_book_ids+1}")
        for link in BeautifulSoup(book_list_page.text, "html.parser").find_all("a"):
            if (
                "ebooks" in link.get("href", "")
                and link["href"].split("/")[2].isdigit()
                and link["href"].split("/")[2] not in page_book_ids
            ):
                page_book_ids.append(link["href"].split("/")[2])
    return page_book_ids


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
