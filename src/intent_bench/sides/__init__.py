from dataclasses import dataclass
from typing import Any

@dataclass
class Side:
    launch: Any
    tolerance: Any
    cuda_graph: bool = False
    device_type: str = 'cuda'
    note: str = ''
    status: str = 'pass'
    owners: Any = None
