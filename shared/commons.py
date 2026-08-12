import argparse


class BookIndex(str):
    __slots__ = ("_source_id",)
    SOURCE = "gutenberg"

    def __new__(cls, source_id):
        return super().__new__(cls, f"{cls.SOURCE}-{int(source_id)}")

    def __init__(self, source_id):
        self._source_id = int(source_id)

    @property
    def source_id(self) -> int:
        return self._source_id

    def __reduce__(self):
        # For botocore errors.
        return (type(self), (self._source_id,))

    @classmethod
    def parse(cls, value: str) -> "BookIndex":
        _, _, source_id = value.rpartition("-")
        return cls(source_id)


def get_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--id", required=True)
    args, _ = parser.parse_known_args()

    if args.platform != BookIndex.SOURCE:
        raise NotImplementedError("Platform not implemented")

    return f"{args.platform}-{args.id}"