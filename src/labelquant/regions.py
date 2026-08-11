from collections.abc import Sequence
from typing import Any

import pandas as pd
from skimage.measure import regionprops_table
from numpy.typing import NDArray


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


def _resolve_region_properties(additional_properties: str | Sequence[str] | None = None,) -> tuple[str, ...]:
    if additional_properties is None:
        return DEFAULT_REGION_PROPERTIES
    
    if isinstance(additional_properties, str):
        additional_properties = (additional_properties,)

    return tuple(dict.fromkeys((*DEFAULT_REGION_PROPERTIES, *additional_properties,)))

