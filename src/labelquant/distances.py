from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table

from labelquant.models import ArrayData


def reference_distance_table(labels: NDArray[Any],
                             references: Mapping[str, ArrayData],
                             *,
                             frame_index: int,
                             ) -> pd.DataFrame:
    """
    Measure each object centroid's pixel distance from every reference channel.

    Empty reference frames produce missing distances. A centroid whose
    containing pixel lies in the reference foreground has distance zero.
    """
    tables = [
        _reference_channel_table(
            labels,
            reference_frame.array,
            ref_label_name=reference_name,
            ref_channel=reference_channel,)
        for reference_name, reference in references.items()
        for reference_channel, reference_frame in _frame_channels(
            reference, frame_index)
    ]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _frame_channels(reference: ArrayData,
                    frame_index: int,
                    ) -> tuple[tuple[str | None, ArrayData], ...]:
    selected_frame = reference.frame(
        0 if reference.frame_count == 1 else frame_index)
    return tuple(selected_frame.iter_channels())


def _reference_channel_table(labels: NDArray[Any],
                             reference: NDArray[Any],
                             *,
                             ref_label_name: str,
                             ref_channel: str | None,
                             ) -> pd.DataFrame:
    if np.any(reference):
        distance_map = distance_transform_edt(~reference.astype(bool))
        properties = regionprops_table(
            label_image=labels,
            intensity_image=distance_map,
            properties=("label",),
            extra_properties=(_centroid_distance,),)
        table = pd.DataFrame(properties).rename(
            columns={"_centroid_distance": "dist_pixel"})
    else:
        labels_present = np.unique(labels)
        labels_present = labels_present[labels_present != 0]
        table = pd.DataFrame({
            "label": labels_present,
            "dist_pixel": np.full(labels_present.size, np.nan),})

    table["ref_label_name"] = ref_label_name
    table["ref_channel"] = ref_channel
    return table


def _centroid_distance(region_mask: NDArray[np.bool_],
                       distance_map: NDArray[np.floating[Any]],
                       ) -> float:
    """Return the distance-map value at the pixel containing an object centroid."""
    centroid = np.mean(np.argwhere(region_mask), axis=0)
    index = tuple(centroid.astype(int))
    return float(distance_map[index])
