from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
import logging

from numpy.typing import NDArray
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArrayData:
    array: NDArray[Any]
    axes: str
    channel_labels: Sequence[str] | None = None
    
    def __post_init__(self) -> None:
        axes = self.axes.upper()
        array = self.array

        if len(axes) != self.array.ndim:
            raise ValueError(f"Axes {axes!r} has {len(axes)} dimensions, but array has {self.array.ndim}.")

        if len(set(axes)) != len(axes):
            raise ValueError(f"Axes must be unique, received {axes!r}.")
        
        channel_labels = (tuple(self.channel_labels)
                          if self.channel_labels is not None 
                          else None)
        
        if "C" in axes:
            c_axis = axes.index("C")
            
            if channel_labels is None:
                raise ValueError("Channel labels must be provided if 'C' is in axes.")
        
            if len(channel_labels) != self.array.shape[c_axis]:
                raise ValueError(f"Number of channel labels ({len(channel_labels)}) does not match the size of the 'C' axis ({self.array.shape[c_axis]}).")
            
            # Move the channel axis to the last postion for consistency
            if c_axis != self.array.ndim - 1:
                array = np.moveaxis(array, c_axis, -1)
                axes = axes.replace("C", "") + "C"
        
        elif channel_labels is not None and len(channel_labels) != 1:
                raise ValueError("An array without a C axis accepts at most one channel label.")
        
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "channel_labels", channel_labels)
        
    @property
    def frame_count(self) -> int:
        """
        Return the number of frames in the array, based on the 'T' axis if present.
        """
        if "T" in self.axes:
            t_axis = self.axes.index("T")
            return self.array.shape[t_axis]
        return 1
    
    @property
    def channel_count(self) -> int:
        """
        Return the number of channels in the array, based on the 'C' axis if present.
        """
        if "C" in self.axes:
            return self.array.shape[-1]
        return 1
    
    def frame(self, index: int) -> ArrayData:
        """
        Return a numpy view of the array corresponding to the specified frame index, based on the 'T' axis if present.
        """
        if "T" not in self.axes:
            if index != 0:
                raise IndexError(f"Array has no 'T' axis, so only index 0 is valid. Received index {index}.")
            return self
        frame_count = self.frame_count
        
        if not 0 <= index < frame_count:
            raise IndexError(f"Frame index {index} is out of bounds for array with '0 to {frame_count-1}' frames.")
        
        time_axis = self.axes.index("T")
        
        selection = [slice(None)] * self.array.ndim
        selection[time_axis] = slice(index, index + 1)
        
        frame_array = self.array[tuple(selection)].squeeze(axis=time_axis)
        frame_axes = self.axes.replace("T", "")
        
        return ArrayData(array=frame_array, axes=frame_axes, channel_labels=self.channel_labels)
    
    def iter_frames(self) -> Iterator[tuple[int, ArrayData]]:
        """
        Iterate over frames in the array, yielding ArrayData for each frame.
        """
        for i in range(self.frame_count):
            yield i, self.frame(i)


    def channel(self, index: int) -> tuple[str | None, ArrayData]:
        """
        Return a numpy view of the array corresponding to the specified channel index, based on the 'C' axis if present.
        """
        if "C" not in self.axes:
            if index != 0:
                raise IndexError(f"Array has no 'C' axis, so only index 0 is valid. Received index {index}.")
            return self.channel_labels[0] if self.channel_labels is not None else None, self
        
        channel_count = self.channel_count
        
        if not 0 <= index < channel_count:
            raise IndexError(f"Channel index {index} is out of bounds for array with '0 to {channel_count-1}' channels.")
        
        if self.channel_labels is None:
            raise RuntimeError("Internal error: an array with a C axis has no channel labels.")
        
        channel_label = self.channel_labels[index]
        channel_array = self.array[..., index]
        channel_axes = self.axes.replace("C", "")
        
        return channel_label, ArrayData(array=channel_array, axes=channel_axes, channel_labels=(channel_label,))
    
    def iter_channels(self) -> Iterator[tuple[str | None, ArrayData]]:
        """
        Iterate over channels in the array, yielding ArrayData for each channel.
        """
        for i in range(self.channel_count):
            yield self.channel(i)
    

class ArrayRole(StrEnum):
    INTENSITY = "intensity"
    OBJECT_LABELS = "object_labels"
    REFERENCE = "reference"
    

@dataclass
class ArrayInputs:
    intensity: ArrayData | None = None
    object_labels: dict[str, ArrayData] = field(default_factory=dict)
    references: dict[str, ArrayData] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.intensity is None:
            return

        for name, labels in self.object_labels.items():
            self._validate_object_labels(name, labels, self.intensity)
    
    def add_array(self, 
                  role: ArrayRole, 
                  array: NDArray[Any],
                  axes: str,
                  *,
                  name: str | None = None,
                  channel_labels: Sequence[str] | None = None) -> None:
        """
        Add an array to the ArrayInputs instance, validating its role and axes.
        
        Args:
            role (ArrayRole): The role of the array (i.e., intensity, object_labels or reference).
            array (NDArray[Any]): The array data to add.
            axes (str): A string representing the axes of the array.
            name (str | None): Name is only optional for intensity arrays, otherwise it is required. Defaults to None.
            channel_labels (Sequence[str] | None, optional): Optional labels for channels if 'C' is in axes. Defaults to None.
        Raises:
            ValueError: If the array's axes are incompatible with its role or if required parameters are missing.
        """
        data = ArrayData(array=array, axes=axes, channel_labels=channel_labels)
        
        if role is ArrayRole.INTENSITY:
            for object_name, labels in self.object_labels.items():
                self._validate_object_labels(object_name, labels, data)
            self.intensity = data
            return
        
        if name is None:
            raise ValueError("Name must be provided for object labels or reference arrays.")
        
        if role is ArrayRole.OBJECT_LABELS:
            if self.intensity is not None:
                self._validate_object_labels(name, data, self.intensity)
            if name in self.object_labels:
                logger.warning(f"Overwriting existing object labels with name '{name}'.")
            self.object_labels[name] = data
        elif role is ArrayRole.REFERENCE:
            if name in self.references:
                logger.warning(f"Overwriting existing reference array with name '{name}'.")
            self.references[name] = data
        else:
            raise ValueError(f"Unknown role: {role}")

    @staticmethod
    def _validate_object_labels(name: str,
                                labels: ArrayData,
                                intensity: ArrayData,
                                ) -> None:
        """
        Validate that object labels can be measured against an intensity array.
        """
        intensity_axes = intensity.axes.removesuffix("C")
        labels_axes = labels.axes.removesuffix("C")

        if intensity_axes != labels_axes:
            raise ValueError(f"Intensity axes {intensity.axes!r} are incompatible with "
                             f"object labels {name!r} axes {labels.axes!r}. After removing "
                             f"channel axes, expected {intensity_axes!r} but received "
                             f"{labels_axes!r}.")

        if intensity.frame_count != labels.frame_count:
            raise ValueError(f"Intensity array has {intensity.frame_count} frames, but object labels {name!r} have {labels.frame_count} frames.")

        intensity_shape = (intensity.array.shape[:-1]
                           if "C" in intensity.axes
                           else intensity.array.shape)
        
        labels_shape = (labels.array.shape[:-1]
                        if "C" in labels.axes
                        else labels.array.shape)

        if intensity_shape != labels_shape:
            raise ValueError(f"Intensity array shape without channels is {intensity_shape}, "
                             f"but object labels {name!r} shape without channels is "
                             f"{labels_shape}.")
