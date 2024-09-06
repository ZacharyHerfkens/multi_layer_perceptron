import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

type arr = NDArray[np.float32]
type Data = list[tuple[arr, arr]]