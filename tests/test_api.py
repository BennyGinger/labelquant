import numpy as np
import pandas as pd
import pytest

from labelquant import ExtractData


def _extractor(*, interval: float | None = None) -> ExtractData:
    image = np.zeros((2, 2, 4, 4), dtype=np.uint16)
    image[:, 0] = 10
    image[:, 1] = 20

    labels = np.zeros((2, 4, 4), dtype=np.uint16)
    labels[:, 1:3, 1:3] = 1

    extractor = ExtractData(interval=interval)
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


def test_quantification_adds_time_in_seconds() -> None:
    dataframe = _extractor(interval=10.0).quantify(workers=1)

    assert dataframe.loc[dataframe["frame"] == 1, "time_sec"].eq(0.0).all()
    assert dataframe.loc[dataframe["frame"] == 2, "time_sec"].eq(10.0).all()


def test_quantification_uses_missing_time_without_interval() -> None:
    dataframe = _extractor().quantify(workers=1)

    assert dataframe["time_sec"].isna().all()


def test_quantification_adds_normalized_reference_distances() -> None:
    extractor = _extractor()
    reference = np.zeros((2, 2, 4, 4), dtype=np.uint8)
    reference[:, 0, 1, 1] = 1
    reference[:, 1, 3, 3] = 1
    extractor.add_array(
        "reference",
        reference,
        "TCYX",
        name="needle",
        channel_labels=("GFP", "RFP"),)

    dataframe = extractor.quantify(workers=1)

    assert len(dataframe) == 8
    assert set(dataframe["ref_label_name"]) == {"needle"}
    assert set(dataframe["ref_channel"]) == {"GFP", "RFP"}
    assert set(dataframe.loc[dataframe["ref_channel"] == "GFP", "dist_pixel"]) == {0.0}
    assert set(dataframe.loc[dataframe["ref_channel"] == "RFP", "dist_pixel"]) == {np.sqrt(8)}


def test_empty_reference_frame_produces_missing_distance() -> None:
    extractor = _extractor()
    reference = np.zeros((2, 4, 4), dtype=np.uint8)
    reference[0, 1, 1] = 1
    extractor.add_array(
        "reference",
        reference,
        "TYX",
        name="needle",
        channel_labels=("GFP",),)

    dataframe = extractor.quantify(workers=1)

    assert dataframe.loc[dataframe["frame"] == 1, "dist_pixel"].eq(0.0).all()
    assert dataframe.loc[dataframe["frame"] == 2, "dist_pixel"].isna().all()


def test_static_reference_is_applied_to_every_frame() -> None:
    extractor = _extractor()
    reference = np.zeros((4, 4), dtype=np.uint8)
    reference[1, 1] = 1
    extractor.add_array(
        "reference",
        reference,
        "YX",
        name="needle",
        channel_labels=("GFP",),)

    dataframe = extractor.quantify(workers=1)

    assert dataframe["dist_pixel"].eq(0.0).all()


def test_reference_arrays_must_be_binary() -> None:
    extractor = _extractor()
    reference = np.zeros((2, 4, 4), dtype=np.uint8)
    reference[0, 1, 1] = 2

    with pytest.raises(ValueError, match="binary values"):
        extractor.add_array(
            "reference", reference, "TYX", name="needle", channel_labels=("GFP",))


def test_reference_array_requires_a_channel_label() -> None:
    extractor = _extractor()
    reference = np.zeros((2, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel label"):
        extractor.add_array("reference", reference, "TYX", name="needle")


def test_parallel_reference_quantification_matches_serial() -> None:
    extractor = _extractor()
    reference = np.zeros((2, 4, 4), dtype=np.uint8)
    reference[:, 1, 1] = 1
    extractor.add_array(
        "reference",
        reference,
        "TYX",
        name="needle",
        channel_labels=("GFP",),)

    serial = extractor.quantify(workers=1)
    parallel = extractor.quantify(workers=2)

    pd.testing.assert_frame_equal(parallel, serial)


@pytest.mark.parametrize("workers", [0, -1, True])
def test_quantify_rejects_invalid_worker_count(workers: int) -> None:
    with pytest.raises(ValueError, match="workers"):
        _extractor().quantify(workers=workers)
