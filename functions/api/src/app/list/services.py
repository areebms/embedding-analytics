from shared.aws import TermTable, get_pipeline_table, get_session


def generate_book_data(fields):
    for item in get_pipeline_table().get_all_entries(fields + ["s3_prefix_models"]):
        if "s3_prefix_models" not in item:
            continue
        item.pop("s3_prefix_models")
        yield item

def get_term_table():
    return TermTable(get_session())

