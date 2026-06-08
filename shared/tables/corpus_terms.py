import os
from datetime import datetime, timezone

from boto3.dynamodb.conditions import Key

from shared.session import get_session
from shared.tables.base import BaseTable

TERM_CORPUS_TABLE = os.getenv("TERM_CORPUS_TABLE")


_corpus_term_table = None


def get_corpus_term_table():
    global _corpus_term_table
    if _corpus_term_table is None:
        _corpus_term_table = CorpusTermTable(get_session())
    return _corpus_term_table


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CorpusTermTable(BaseTable):
    """Corpus-wide aggregation per term.

    Single partition (#ALL) to enable a fast query for all terms across
    all books. Each row records which books contain the term and the
    total occurrence count across books.

    Row shape:
        partition (PK, str = "#ALL")
        term (SK, str): "labour"
        book_ids (set[str]): {"gutenberg-3300", "gutenberg-33310"}
        book_count (int)
    """

    PARTITION = "#ALL"

    def __init__(self, session):
        super().__init__(session, TERM_CORPUS_TABLE)

    def get_term(self, term):
        return super().get_entry({"partition": self.PARTITION, "term": term})

    def list_all_terms(self):
        return self.list_all(KeyConditionExpression=Key("partition").eq(self.PARTITION))

    def add_book(self, term, book_id):
        self.table.update_item(
            Key={"partition": self.PARTITION, "term": term},
            UpdateExpression=("ADD book_ids :bid SET updated_at = :now"),
            ExpressionAttributeValues={":bid": {book_id}, ":now": utc_now()},
        )

    def remove_book(self, term, book_id):
        self.table.update_item(
            Key={"partition": self.PARTITION, "term": term},
            UpdateExpression=("DELETE book_ids :bid SET updated_at = :now"),
            ExpressionAttributeValues={":bid": {book_id}, ":now": utc_now()},
        )

    def remove_book_terms(self, book_id, terms):
        """Remove book_id from multiple term rows using batch get + batch write."""
        if not terms:
            return

        current_term_data = {}
        unprocessed_keys = [
            {"partition": self.PARTITION, "term": term} for term in terms
        ]
        while True:
            response = self.dynamodb.batch_get_item(
                RequestItems={TERM_CORPUS_TABLE: {"Keys": unprocessed_keys}}
            )
            for item in response["Responses"].get(TERM_CORPUS_TABLE, []):
                current_term_data[item["term"]] = item

            if TERM_CORPUS_TABLE not in response["UnprocessedKeys"]:
                break

            unprocessed_keys = response["UnprocessedKeys"][TERM_CORPUS_TABLE]["Keys"]

        now = utc_now()
        with self.table.batch_writer() as batch:
            for term, term_data in current_term_data.items():
                term_data["book_ids"].discard(book_id)
                if term_data["book_ids"]:
                    term_data["book_ids"] = term_data["book_ids"]
                    term_data["updated_at"] = now
                    batch.put_item(Item=item)
                else:
                    batch.delete_item(Key={"partition": self.PARTITION, "term": term})
