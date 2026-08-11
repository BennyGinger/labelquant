from collections.abc import Sequence
from typing import Any
from dataclasses import dataclass, field

from labelquant.formatting import format_region_table
from labelquant.models import ArrayInputs, ArrayRole
from numpy.typing import NDArray
import pandas as pd

from labelquant.regions import extract_region_frame


@dataclass
class ExtractData:
    array_data: ArrayInputs = field(default_factory=ArrayInputs)
    
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
            name (str | None): Name is only optional for intensity arrays, otherwise it is required. Defaults to None.
            channel_labels (Sequence[str] | None, optional): Optional labels for channels if 'C' is in axes. Defaults to None.
        """
        self.array_data.add_array(role=role, array=array, axes=axes, name=name, channel_labels=channel_labels)
        
    def quantify(self, additional_properties: str | Sequence[str] | None = None) -> pd.DataFrame:
        """
        Quantify the object labels in the intensity array and return a DataFrame of region properties.
        
        Default region properties include: area, centroid, intensity_mean, label, perimeter, and solidity. Additional properties can be specified using the `additional_properties` parameter.
        """
        intensity_array = self.array_data.intensity
        
        if intensity_array is None:
            raise ValueError("Intensity array is not set. Please add an intensity array before quantification.")
        
        object_labels_dict = self.array_data.object_labels
        
        if not object_labels_dict:
            raise ValueError("No object labels are set. Please add at least one object labels array before quantification.")
        
        results: list[pd.DataFrame] = []
        for name, labels_array in object_labels_dict.items():
            for object_channel, label_stack in labels_array.iter_channels():
                for frame_index, label_frame in label_stack.iter_frames():
                    image_frame = intensity_array.frame(frame_index)
                    
                    frame_df = extract_region_frame(image=image_frame.array, 
                                                    labels=label_frame.array,
                                                    additional_properties=additional_properties)
                    
                    frame_df = format_region_table(df=frame_df,
                                                   channel_labels=image_frame.channel_labels,
                                                   label_axes=label_frame.axes)
                    
                    frame_df['frame'] = frame_index + 1
                    frame_df['object_name'] = name
                    frame_df['object_channel'] = object_channel
                    
                    results.append(frame_df)

        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)
