#!/usr/bin/env python3

"""
Battery is described by its maximum capacity.
The current capacity describes the temporary state.
"""

from functools import cached_property
from .constants import (
    CHARGE_THRESHOLD,
    CONVERSION_LOSS,
    EPS,
    RETENTION_RATE,
    C_RATE,
    DISCHARGE_THRESHOLD,
)
import pandas as pd
from typing import List


class Battery:
    capacity: float
    _current_cap: float
    _capacity_curve: List[float]
    _timestep_duration: float

    def __init__(self, capacity: float, timestep_duration: float):
        assert capacity >= 0, "Battery Capacity not allowed to be negative."
        self.capacity = capacity
        self._current_cap = DISCHARGE_THRESHOLD * capacity
        self._capacity_curve = []
        self._timestep_duration = timestep_duration

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
        # some energy is lost when converting
        # don't charge faster than C_RATE
        # don't charge beyond charging threshold
        maxChargeAmount: float = max(0.0, self.maxAllowedCharge - self._current_cap)
        cRateLimit: float = self.getCRateLimit()
        maxChargeAmount = min(maxChargeAmount, cRateLimit)
        return maxChargeAmount * (1 / (1 - CONVERSION_LOSS))

    def getCRateLimit(self) -> float:
        return C_RATE * self._timestep_duration * self.capacity

    def dischargeAmount(self) -> float:
        # some energy is lost when converting
        # only discharge to threshold (15% by default)
        # discharge at rate no faster than given by c_rate
        maxDischargeAmount: float = max(0.0, self._current_cap - self.minAllowedCharge)
        cRateLimit: float = C_RATE * self._timestep_duration * self.capacity
        maxDischargeAmount = min(maxDischargeAmount, cRateLimit)
        return maxDischargeAmount * (1 - CONVERSION_LOSS)

    def reset(self) -> None:
        self._capacity_curve = []
        self._current_cap = DISCHARGE_THRESHOLD * self.capacity

    @cached_property
    def minAllowedCharge(self) -> float:
        return DISCHARGE_THRESHOLD * self.capacity

    @cached_property
    def maxAllowedCharge(self) -> float:
        return CHARGE_THRESHOLD * self.capacity

    def getCapacityCurve(self) -> pd.Series:
        return pd.Series(self._capacity_curve)

    def _timestep(self) -> None:
        """
        Executes bookkeeping operations that signify a timestep was executed.
        """
        self._current_cap = min(self.capacity, max(0, self._current_cap))
        self._current_cap *= 1 - ((1 - RETENTION_RATE) * self._timestep_duration)
        self._capacity_curve.append(self._current_cap)
