from typing import Dict, Hashable, Tuple, Union

DimsT = Tuple[Union[Hashable, None]]
ShapeT = Tuple[Union[int, None]]
ChunksT = Union[bool, Dict[Hashable, Union[int, None]]]
