import json
import os

from shared.aws import PipelineTable, get_session
from shared.commons import get_index

KVECTORS_TRAINED = os.environ.get("KVECTORS_TRAINED", 2)


def initiate_train_vectors(index):
    client = get_session().client("lambda")

    passed_count = 0
    for _ in range(KVECTORS_TRAINED):

        status_code = client.invoke(
            FunctionName="train-kvector",
            InvocationType="Event",
            Payload=json.dumps({"index": index}).encode("utf-8"),
        ).get("StatusCode")
        if status_code == 202:
            passed_count += 1

    return passed_count, KVECTORS_TRAINED


if __name__ == "__main__":

    index = get_index()
    session = get_session()
    table = PipelineTable(session)

    item = table.get(index, expression="platform_data,s3_token_lemmas_key")

    s3_token_lemmas_key = item.get("s3_token_lemmas_key")
    if not s3_token_lemmas_key:
        print(f"{index} has not been tokenized.")
        exit()

    initiate_train_vectors(index)