from typing import Dict, Tuple, Union

import numpy as np

DTypeLike = np.typing.DTypeLike
DimsT = Tuple[Union[str, None]]
ShapeT = Tuple[Union[int, None]]
ChunksT = Union[bool, Dict[str, Union[int, None]]]
