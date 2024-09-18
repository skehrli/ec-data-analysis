#!/usr/bin/env python3

import networkx as nx

class Battery:
    _capacity: float
    _
    _current_cap: float

    def __init__(self, capacity: float):
        self._capacity = capacity
        self._current_cap = 0.0

    def charge(self, N: nx.DiGraph) -> tuple[float, dict[str, dict[str, float]]]:
        pass

    def discharge(self, N: nx.DiGraph) -> tuple[float, dict[str, dict[str, float]]]:
        pass

    def _isChargeable(self) -> bool:
        return True

    def _isDischargeable(self) -> bool:
        # pay attention - even if this returns true,
        # the capacity shouldn't fall under the limit
        return True
