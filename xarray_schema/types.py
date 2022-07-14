from typing import Dict, Tuple, Union

from numpy.typing import DTypeLike  # noqa: F401

DimsT = Tuple[Union[str, None]]
ShapeT = Tuple[Union[int, None]]
ChunksT = Union[bool, Dict[str, Union[int, None]]]
