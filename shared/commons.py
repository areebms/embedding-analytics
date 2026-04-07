import argparse

def get_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--id", required=True)
    args, _ = parser.parse_known_args()

    if args.platform != "gutenberg":
        raise NotImplementedError("Platform not implemented")

    return f"{args.platform}-{args.id}"