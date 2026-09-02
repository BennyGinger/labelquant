from collections.abc import Sequence
from typing import Any

import pandas as pd
from skimage.measure import regionprops_table
from numpy.typing import NDArray

from labelquant.formatting import format_region_table
from labelquant.distances import reference_distance_table
from labelquant.models import ArrayData


DEFAULT_REGION_PROPERTIES = ('area',
                             'centroid',
                             'intensity_mean',
                             'label',
                             'perimeter',
                             'solidity')


def extract_region_frame(image: NDArray[Any],
                         labels: NDArray[Any],
                         *,
                         additional_properties: str | Sequence[str] | None = None,
                         ) -> pd.DataFrame:
    """Extract region properties from a labeled image and return as a DataFrame."""
    properties = _resolve_region_properties(additional_properties)
    
    props = regionprops_table(
        label_image=labels, 
        intensity_image=image,
        properties=properties,
        separator='_')
    return pd.DataFrame(props)


def quantify_region_frame(
    frame_index: int,
    intensity: ArrayData,
    labels: ArrayData,
    object_name: str,
    object_channel: str | None,
    additional_properties: str | Sequence[str] | None,
    references: dict[str, ArrayData],
) -> pd.DataFrame:
    """
    Prepare and quantify one complete frame of an object-label array.
    """
    image_frame = intensity.frame(frame_index)
    label_frame = labels.frame(frame_index)

    frame_df = extract_region_frame(
        image=image_frame.array,
        labels=label_frame.array,
        additional_properties=additional_properties,
    )
    frame_df = format_region_table(
        df=frame_df,
        channel_labels=image_frame.channel_labels,
        label_axes=label_frame.axes,
    )

    reference_df = reference_distance_table(
        label_frame.array,
        references,
        frame_index=frame_index,)
    if not reference_df.empty:
        frame_df = frame_df.merge(reference_df, on="label", how="left")

    frame_df["frame"] = frame_index + 1
    frame_df["object_name"] = object_name
    frame_df["object_channel"] = object_channel
    return frame_df


def _resolve_region_properties(additional_properties: str | Sequence[str] | None = None,) -> tuple[str, ...]:
    if additional_properties is None:
        return DEFAULT_REGION_PROPERTIES
    
    if isinstance(additional_properties, str):
        additional_properties = (additional_properties,)

    return tuple(dict.fromkeys((*DEFAULT_REGION_PROPERTIES, *additional_properties,)))
