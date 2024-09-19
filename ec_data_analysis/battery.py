#!/usr/bin/env python3

"""
Battery is described by its maximum capacity and retention rate (between 0 and 1) of capacity between
timesteps. The current capacity describes the temporary state.
"""

from .constants import CONVERSION_LOSS, EPS
import pandas as pd
from typing import List


class Battery:
    _capacity: float
    _retention_rate: float
    _current_cap: float
    _capacity_curve: List[float]

    def __init__(self, capacity: float, retention_rate: float):
        assert capacity >= 0, "Battery Capacity not allowed to be negative."
        self._capacity = capacity
        self._current_cap = 0.0
        self._retention_rate = retention_rate
        self._capacity_curve = []

    def charge(self, amount: float) -> None:
        if amount - self.chargeAmount() > EPS:
            print(
                f"ERROR: chargeamount > possible topup by {amount - self.chargeAmount()}"
            )
        amount *= 1 - CONVERSION_LOSS  # conversion to battery loses some power
        self._current_cap += amount
        self._timestep()

    def discharge(self, amount: float) -> None:
        if amount - self.dischargeAmount() > EPS:
            print(
                f"ERROR: dischargeamount > remaining charge by {amount - self.dischargeAmount()}"
            )
        amount *= 1 / (1 - CONVERSION_LOSS)
        self._current_cap -= amount
        self._timestep()

    def chargeAmount(self) -> float:
        return (self._capacity - self._current_cap) * (1 / (1 - CONVERSION_LOSS))

    def dischargeAmount(self) -> float:
        # some energy is lost when converting
        return self._current_cap * (1 - CONVERSION_LOSS)

    def getCapacityCurve(self) -> pd.Series:
        return pd.Series(self._capacity_curve)

    def _timestep(self) -> None:
        """
        Executes bookkeeping operations that signify a timestep was executed.
        """
        self._current_cap = min(self._capacity, max(0, self._current_cap))
        self._current_cap *= self._retention_rate
        self._capacity_curve.append(self._current_cap)
