import copy
import os
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv
from gensim.models import KeyedVectors

from shared.session import get_session


load_dotenv(Path(__file__).parents[3] / ".env")


S3_BUCKET = os.environ.get("S3_BUCKET")
S3_TEST_DATA_PREFIX = os.environ.get("S3_TEST_DATA_PREFIX")


@pytest.fixture(scope="session")
def s3_test_data_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("s3_data")
    session = get_session()
    bucket = session.resource("s3").Bucket(S3_BUCKET)
    for obj in bucket.objects.filter(Prefix=S3_TEST_DATA_PREFIX):
        dest = tmp_dir / Path(obj.key).name
        bucket.download_file(obj.key, str(dest))
    return tmp_dir


@pytest.fixture(scope="session")
def kvector_stack(s3_test_data_dir):
    models = sorted(s3_test_data_dir.glob("*.model"))
    return [KeyedVectors.load(str(p)) for p in models]


@pytest.fixture(scope="session")
def alignment_result(kvector_stack):
    from procrustes_utils import gradient_descent_alignment, build_centroid_kvector

    stack = copy.deepcopy(kvector_stack)
    terms = list(stack[0].key_to_index)

    centroid_vectors, residuals, mean_disparity, _ = gradient_descent_alignment(
        terms, stack
    )
    counts = np.array([stack[0].get_vecattr(t, "count") for t in terms])
    centroid = build_centroid_kvector(terms, counts, residuals, centroid_vectors)

    return mean_disparity, centroid
