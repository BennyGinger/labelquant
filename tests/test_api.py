import numpy as np
import pandas as pd
import pytest

from labelquant import ExtractData


def _extractor() -> ExtractData:
    image = np.zeros((2, 2, 4, 4), dtype=np.uint16)
    image[:, 0] = 10
    image[:, 1] = 20

    labels = np.zeros((2, 4, 4), dtype=np.uint16)
    labels[:, 1:3, 1:3] = 1

    extractor = ExtractData()
    extractor.add_array(
        "intensity",
        image,
        "TCYX",
        channel_labels=("GFP", "RFP"),
    )
    extractor.add_array(
        "object_labels",
        labels,
        "TYX",
        name="tracking",
    )
    return extractor


def test_parallel_quantification_matches_serial() -> None:
    serial = _extractor().quantify(workers=1)
    parallel = _extractor().quantify(workers=2)

    pd.testing.assert_frame_equal(parallel, serial)


@pytest.mark.parametrize("workers", [0, -1, True])
def test_quantify_rejects_invalid_worker_count(workers: int) -> None:
    with pytest.raises(ValueError, match="workers"):
        _extractor().quantify(workers=workers)
