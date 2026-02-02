import argparse

def get_index():
    parser = argparse.ArgumentParser(
        description="Scrape and upload Gutenberg data to AWS."
    )
    parser.add_argument("--platform-name", required=True)
    parser.add_argument("--platform-id", required=True)
    args = parser.parse_args()

    if args.platform_name != "gutenberg":
        raise NotImplementedError("Platform not implemented")

    return f"{args.platform_name}-{args.platform_id}"