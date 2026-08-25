from collections.abc import Sequence
from typing import Any
from dataclasses import dataclass, field

from labelquant.models import ArrayInputs, ArrayRole
from joblib import Parallel, delayed
from numpy.typing import NDArray
import pandas as pd

from labelquant.regions import quantify_region_frame


# Arrays larger than this are memory-mapped once and shared read-only between joblib workers.
MEMMAP_THRESHOLD = "10M"


@dataclass
class ExtractData:
    """Arrays and acquisition timing used for region quantification.

    Attributes:
        array_data: Intensity, object-label, and reference arrays to quantify.
        interval: Time between consecutive frames in seconds. When unavailable,
            the output ``time_sec`` column contains missing values.
    """

    array_data: ArrayInputs = field(default_factory=ArrayInputs)
    interval: float | None = None

    def add_array(self,
                  role: ArrayRole,
                  array: NDArray[Any],
                  axes: str,
                  *,
                  name: str | None = None,
                  channel_labels: Sequence[str] | None = None) -> None:
        """Add an array to the ExtractData instance.

        Args:
            role (ArrayRole): The role of the array (i.e., intensity, object_labels or reference).
            array (NDArray[Any]): The array data to add.
            axes (str): A string representing the axes of the array.
            name (str | None): Name is only optional for intensity arrays, otherwise it is required. It should be unique name to identify the object labels (it would be used on the output sheet, e.g. 'tracking', 'cyto'...), Defaults to None.
            channel_labels (Sequence[str] | None, optional): Optional labels for channels if 'C' is in axes. Defaults to None.
        """
        self.array_data.add_array(role=role, array=array, axes=axes, name=name, channel_labels=channel_labels)

    def quantify(
        self,
        additional_properties: str | Sequence[str] | None = None,
        *,
        workers: int = 1,
    ) -> pd.DataFrame:
        """
        Quantify the object labels in the intensity array and return a DataFrame of region properties.

        Default region properties include: area, centroid, intensity_mean, label, perimeter, and solidity. Additional properties can be specified using the `additional_properties` parameter.

        Args:
            additional_properties: Additional scikit-image region properties to extract.
            workers: Number of frame processes. A value of 1 runs serially.

        The returned table contains a one-based ``frame`` column and a
        zero-based ``time_sec`` column derived from ``interval``.
        """
        if isinstance(workers, bool) or workers < 1:
            raise ValueError("workers must be an integer greater than or equal to 1.")

        intensity_array = self.array_data.intensity

        if intensity_array is None:
            raise ValueError("Intensity array is not set. Please add an intensity array before quantification.")
        
        object_labels_dict = self.array_data.object_labels

        if not object_labels_dict:
            raise ValueError("No object labels are set. Please add at least one object labels array before quantification.")
        
        jobs = (
            (frame_index,
             intensity_array,
             label_stack,
             name,
             object_channel,
             additional_properties,
            )
            for name, labels_array in object_labels_dict.items()
            for object_channel, label_stack in labels_array.iter_channels()
            for frame_index in range(label_stack.frame_count)
        )

        if workers == 1:
            results = [quantify_region_frame(*job) for job in jobs]
        else:
            results = Parallel(
                n_jobs=workers,
                backend="loky",
                max_nbytes=MEMMAP_THRESHOLD,
                mmap_mode="r",
            )(
                delayed(quantify_region_frame)(*job)
                for job in jobs
            )

        if not results:
            return pd.DataFrame()

        dataframe = pd.concat(results, ignore_index=True)

        if self.interval is None:
            dataframe["time_sec"] = pd.NA
        else:
            dataframe["time_sec"] = (
                dataframe["frame"] - 1
            ) * self.interval

        return dataframe




if __name__ == "__main__":
    from fits_io import FitsIO
    from time import time


    time_0 = time()
    parent = "/media/ben/Analysis/Python/Images/zymosan/zym_chamber_500k_WT_HoxB8_CalB630_001-MaxIP_s1/"

    img_path = "/media/ben/Analysis/Python/Images/zymosan/zym_chamber_500k_WT_HoxB8_CalB630_001-MaxIP_s1/fits_array.tif"
    img_reader = FitsIO.from_path(img_path)
    img = img_reader.get_array()

    mask_path = "/media/ben/Analysis/Python/Images/zymosan/zym_chamber_500k_WT_HoxB8_CalB630_001-MaxIP_s1/fits_track.tif"
    mask_reader = FitsIO.from_path(mask_path)
    mask = mask_reader.get_array()
    print(f"Loaded image and mask in {time() - time_0:.2f} seconds.")
    time_start = time()
    
    extractor = ExtractData()

    extractor.add_array(role="intensity", array=img.array, axes=img.axes, channel_labels=img_reader.channel_labels)
    extractor.add_array(role="object_labels", array=mask.array, axes=mask.axes, name="tracking", channel_labels=mask_reader.channel_labels)
    print(f"Added arrays to ExtractData in {time() - time_start:.2f} seconds.")
    time_start = time()
    
    extracted_df = extractor.quantify(workers=8)
    print(f"Quantified {len(extracted_df)} regions in {time() - time_start:.2f} seconds.")
    time_start = time()
    
    extracted_df.to_csv(parent + "quantification.csv", index=False)
    print(f"Saved quantification to CSV in {time() - time_start:.2f} seconds.")
    print(f"Quantification completed in {time() - time_0:.2f} seconds.")
