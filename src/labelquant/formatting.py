from __future__ import annotations

from collections.abc import Sequence
import re

import pandas as pd


_INTENSITY_COLUMN = re.compile(r"^intensity_mean_(\d+)$")


def format_region_table(df: pd.DataFrame,
                        *,
                        channel_labels: Sequence[str] | None,
                        label_axes: str,
                        ) -> pd.DataFrame:
    """
    Normalize columns produced by ``regionprops_table``.
    """
    result = df.copy()
    result = rename_centroid_columns(result, label_axes=label_axes)
    return reshape_intensity_columns(result, channel_labels=channel_labels)


def rename_centroid_columns(df: pd.DataFrame,
                            *,
                            label_axes: str,
                            ) -> pd.DataFrame:
    """
    Rename positional centroid columns using their spatial axis names.
    """
    spatial_axes = [axis.lower()
                    for axis in label_axes.upper()
                    if axis not in {"T", "C"}]

    rename = {f"centroid_{index}": f"centroid_{axis}"
              for index, axis in enumerate(spatial_axes)
              if f"centroid_{index}" in df.columns}

    if not rename:
        return df

    return df.rename(columns=rename)


def reshape_intensity_columns(df: pd.DataFrame,
                              *,
                              channel_labels: Sequence[str] | None,
                              ) -> pd.DataFrame:
    """
    Convert regionprops intensity columns to long format when needed.
    """
    indexed_columns: list[tuple[int, str]] = []

    for column in df.columns:
        match = _INTENSITY_COLUMN.fullmatch(column)
        if match is not None:
            indexed_columns.append((int(match.group(1)), column))

    indexed_columns.sort(key=lambda item: item[0])
    intensity_columns = [column for _, column in indexed_columns]

    has_intensity_mean = "intensity_mean" in df.columns
    has_intensity_channel = "intensity_channel" in df.columns

    if not has_intensity_mean and not intensity_columns:
        return df

    if has_intensity_mean and has_intensity_channel and not intensity_columns:
        return df

    if has_intensity_channel and intensity_columns:
        raise ValueError("Cannot reshape indexed intensity columns because intensity_channel already exists.")

    if has_intensity_mean:
        if channel_labels is not None and len(channel_labels) != 1:
            raise ValueError("A single intensity_mean column requires zero or one channel label.")

        channel_name = (channel_labels[0] if channel_labels is not None else None)

        result = df.copy()
        loc = result.columns.get_loc("intensity_mean")
        if isinstance(loc, slice):
            loc = loc.start
        result.insert(int(loc),
                      "intensity_channel",
                      channel_name)
        return result

    if channel_labels is None:
        raise ValueError("Channel labels are required for multichannel intensity columns.")

    if len(channel_labels) != len(intensity_columns):
        raise ValueError(f"Found {len(intensity_columns)} intensity columns but received {len(channel_labels)} channel labels.")

    channel_mapping = dict(zip(intensity_columns, channel_labels,
                               strict=True))
    identifier_columns = [column 
                          for column in df.columns 
                          if column not in intensity_columns]

    result = df.melt(id_vars=identifier_columns,
                     value_vars=intensity_columns,
                     var_name="_intensity_column",
                     value_name="intensity_mean",)
    
    result["intensity_channel"] = result.pop("_intensity_column").map(channel_mapping)
    return result
