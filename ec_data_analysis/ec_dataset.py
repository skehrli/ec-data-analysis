#!/usr/bin/env python3

"""
ec_dataset.py

This module contains the ECDataset class, which is used to manage and manipulate
datasets for energy communities.

TODO add this documentation
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from typing import List
from typing import Optional
from constants import DECIMAL_POINTS
from constants import N_BINS
from functools import cached_property
from market_solution import MarketSolution


class ECDataset:
    # The 4 core DataFrames
    production: pd.DataFrame
    consumption: pd.DataFrame
    supply: pd.DataFrame
    demand: pd.DataFrame

    # Dimensions of the above DataFrames: rows/columns
    numParticipants: int
    numTimesteps: int

    # capacity of the fictional battery
    batteryCapacity: int
    # solutions to the flow problem per time interval
    marketSolutions: List[MarketSolution] = []
    # optional output pdf to write to
    outputFile: Optional[PdfPages]

    def __init__(
        self,
        production: pd.DataFrame,
        consumption: pd.DataFrame,
        batteryCapacity: int = 0,
        outputFile: Optional[PdfPages] = None,
    ) -> None:
        """
        Initialize the data class with production and consumption DataFrames.

        :param production: DataFrame containing production data. production[i][j] is the production in w/h of member j in time interval i.
        :param consumption: DataFrame containing consumption data. consumption[i][j] is the usage in w/h of member j in time interval i.

        :ivar production: DataFrame with production[i][j] being the production in w/h of member j in time interval i.
        :ivar consumption: DataFrame with consumption[i][j] being the usage in w/h of member j in time interval i.
        :ivar supply: DataFrame with supply[i][j] being the amount member j is selling in interval i.
        :ivar demand: DataFrame with demand[i][j] being the amount member j is buying in interval i.
        """
        assert (
            production.shape == consumption.shape
        ), f"production and consumption have unequal shapes {production.shape} and {consumption.shape}"
        self.numTimesteps, self.numParticipants = production.shape
        production.columns = range(self.numParticipants)
        consumption.columns = range(self.numParticipants)
        self.production = production
        self.consumption = consumption
        self.supply = (production - consumption).clip(lower=0)
        self.demand = (consumption - production).clip(lower=0)
        self.batteryCapacity = batteryCapacity
        self.outputFile = outputFile
        df: pd.DataFrame = self.supply - self.demand
        for t in range(self.numTimesteps):
            self.marketSolutions.append(MarketSolution(df.iloc[t]))

    def changeBatteryCapacity(self, newCapacity: int) -> None:
        self.batteryCapacity = newCapacity
        df: pd.DataFrame = self.supply - self.demand
        for t in range(self.numTimesteps):
            self.marketSolutions.append(MarketSolution(df.iloc[t]))

    def printKeyStats(self) -> None:
        self._print(
            "Nr of overproduction vs overconsumption datapoints",
            self._compareProductionWithConsumption,
        )
        self._printPercent(
            "Consumption covered by own production",
            (self.getConsumptionVolume - self.getDemandVolume)
            / self.getConsumptionVolume,
        )
        self._printPercent(
            "Consumption covered by trades",
            self.getTradingVolume / self.getConsumptionVolume,
        )
        self._printPercent(
            "Consumption covered by battery",
            self.getDischargeVolume / self.getConsumptionVolume,
        )
        self._printPercent(
            "Demand covered by trades", self.getTradingVolume / self.getDemandVolume
        )
        self._printPercent(
            "Percentage of supply sold", self.getTradingVolume / self.getSupplyVolume
        )

    def visualizeSellRatioDistribution(self) -> None:
        """
        Make a bar chart visualizing the distribution of the ratio of sold supply.
        """
        sellRatioVec: pd.Series = self.getSellVolumePerMember / self.getSupplyPerMember
        sellRatioVec = sellRatioVec.replace([np.inf], 0)

        # Define bin edges to cover the range from 0 to 1
        bin_edges = np.linspace(0, 1, N_BINS + 1)

        # Discretize the data into bins
        binned_data = pd.cut(sellRatioVec, bins=bin_edges)

        # Create an IntervalIndex from bin edges
        interval_index = pd.IntervalIndex.from_breaks(bin_edges)

        # Compute histogram counts and reindex to include all bins
        hist = binned_data.value_counts(sort=False).reindex(
            interval_index, fill_value=0
        )

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black")
        plt.xlabel("Ratio Sold")
        plt.ylabel("Number of Community Members")
        plt.title("Ratio of Sold Supply per Member")
        plt.grid(True)
        self._show()

    def _show(self) -> None:
        if self.outputFile is not None:
            self.outputFile.savefig()
        else:
            plt.show()

    def visualizeBuyRatioDistribution(self) -> None:
        """
        Make a bar chart visualizing the distribution of the ratio of purchased demand.
        """
        buyRatioVec: pd.Series = self.getBuyVolumePerMember / self.getDemandPerMember
        buyRatioVec = buyRatioVec.replace([np.inf], 0)

        # Define bin edges to cover the range from 0 to 1
        bin_edges = np.linspace(0, 1, N_BINS + 1)

        # Discretize the data into bins
        binned_data = pd.cut(buyRatioVec, bins=bin_edges)

        # Create an IntervalIndex from bin edges
        interval_index = pd.IntervalIndex.from_breaks(bin_edges)

        # Compute histogram counts and reindex to include all bins
        hist = binned_data.value_counts(sort=False).reindex(
            interval_index, fill_value=0
        )

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black")
        plt.xlabel("Ratio Purchased")
        plt.ylabel("Number of Community Members")
        plt.title("Ratio of Purchase Volume / Demand Volume per Member")
        plt.grid(True)
        self._show()

    def _printPercent(self, description: str, num: float) -> None:
        """
        num should be a ratio.
        """
        self._print(description + " (%)", round(num * 100, DECIMAL_POINTS))

    def _print(self, description: str, num: object) -> None:
        print(f"{description}: {num}")

    @cached_property
    def getTradingVolume(self) -> float:
        """
        Returns the overall trading volume over the timeframe of the dataset.
        """
        return sum(sol.tradingVolume for sol in self.marketSolutions)

    @cached_property
    def getDischargeVolume(self) -> float:
        """
        Returns the overall discharging volume over the timeframe of the dataset.
        """
        return sum(sol.chargeAmount for sol in self.marketSolutions)

    @cached_property
    def getDemandVolume(self) -> float:
        """
        Returns the overall demand on the market over the timeframe of the dataset.
        """
        return self.demand.sum().sum()

    @cached_property
    def getSupplyVolume(self) -> float:
        """
        Returns the overall demand on the market over the timeframe of the dataset.
        """
        return self.supply.sum().sum()

    @cached_property
    def getConsumptionVolume(self) -> float:
        """
        Returns the overall consumption of all participants over the timeframe of the dataset.
        """
        return self.consumption.sum().sum()

    @cached_property
    def getDemandPerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall demand.
        """
        return self.demand.sum()

    @cached_property
    def getSupplyPerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall supply.
        """
        return self.supply.sum()

    @cached_property
    def getSellVolumePerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall sell volume.
        """
        return pd.Series(
            {
                i: sum(sol.getQtySoldForMember(i) for sol in self.marketSolutions)
                for i in range(self.numParticipants)
            }
        )

    @cached_property
    def getBuyVolumePerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall buy volume.
        """
        return pd.Series(
            {
                i: sum(sol.getQtyPurchasedForMember(i) for sol in self.marketSolutions)
                for i in range(self.numParticipants)
            }
        )

    @cached_property
    def getChargeVolumePerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall battery charging volume.
        """
        return pd.Series(
            {
                i: sum(sol.getQtyChargedForMember(i) for sol in self.marketSolutions)
                for i in range(self.numParticipants)
            }
        )

    @cached_property
    def getDischargeVolumePerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall battery discharge volume.
        """
        return pd.Series(
            {
                i: sum(sol.getQtyDischargedForMember(i) for sol in self.marketSolutions)
                for i in range(self.numParticipants)
            }
        )

    @cached_property
    def _compareProductionWithConsumption(self) -> tuple[int, int]:
        return (self.supply > 0).sum().sum(), (self.demand > 0).sum().sum()
