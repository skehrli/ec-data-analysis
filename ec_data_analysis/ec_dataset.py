#!/usr/bin/env python3

"""
ec_dataset.py

This module contains the ECDataset class, which is used to manage and manipulate
datasets for energy communities.

TODO add this documentation
"""

from .battery import Battery
from .constants import DECIMAL_POINTS, RETENTION_RATE, N_BINS
from .report import Report
from .market_solution import MarketSolution
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy import floating
from numpy.typing import NDArray
from typing import List, Optional, Any
from functools import cached_property
import subprocess


class ECDataset:
    # The 4 core DataFrames
    production: pd.DataFrame
    consumption: pd.DataFrame
    supply: pd.DataFrame
    demand: pd.DataFrame

    # Dimensions of the above DataFrames: rows/columns
    numParticipants: int
    numTimesteps: int

    # fictional battery
    battery: Optional[Battery]
    # solutions to the flow problem per time interval
    marketSolutions: List[MarketSolution] = []

    # used to compile a pdf report
    report: Optional[Report]
    _uid: int

    def __init__(
        self,
        production: pd.DataFrame,
        consumption: pd.DataFrame,
        battery: Optional[Battery] = None,
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
        self.battery = battery
        self.report = None
        df: pd.DataFrame = self.supply - self.demand
        for t in range(self.numTimesteps):
            self.marketSolutions.append(MarketSolution(df.iloc[t]))
        self._uid = 0

    def changeBattery(self, newBattery: Battery) -> None:
        self.battery = newBattery
        for t in range(self.numTimesteps):
            self.marketSolutions[t].computeWithBattery(newBattery)

    def createReport(self, capacities: np.ndarray) -> None:
        report: Report = Report("EC Report")
        self.report = report

        report.addHeading("Key Statistics", 1)
        self.printKeyStats()

        report.addHeading("Sell Ratio", 2)
        report.dump(
            "The following figure plots the ratio of sold energy on the local market to the supply per community member.\nThis is to get a sense of how fair the distribution for sellers is."
        )
        report.putFig("Sell Ratio", self.visualizeSellRatioDistribution())

        report.addHeading("Buy Ratio", 2)
        report.dump(
            "The following figure plots the ratio of purchased energy on the local market to the demand per community member.\nThis is to get a sense of how fair the distribution for buyers is."
        )
        report.putFig("Buy Ratio Distribution", self.visualizeBuyRatioDistribution())

        report.addHeading("Statistics With Battery", 1)
        report.dump(
            "The following section compares how batteries of different capacities affect the community."
        )

        for maxCap in capacities:
            battery: Battery = Battery(maxCap, RETENTION_RATE)
            self.changeBattery(battery)

            report.addHeading(f"Battery Capacity {maxCap} kw/h", 2)
            self.printBatteryStats()

            report.addHeading(
                "Ratio of energy obtained from battery of overall demand", 3
            )
            report.dump(
                "The following figure plots the ratio of energy obtained from the battery of the overall bought energy per community member."
            )
            report.putFig(
                "Battery Discharge Ratio", self.visualizeDischargeRatioDistribution()
            )

            report.addHeading(
                "Ratio of energy sold to battery of overall sold energy", 3
            )
            report.dump(
                "The following figure plots the ratio of energy sold to the battery of the overall sold energy per community member."
            )
            report.putFig(
                "Battery Charge Ratio", self.visualizeChargeRatioDistribution()
            )

            report.addHeading(
                "Ratio of energy sold to battery of overall sold energy", 3
            )
            report.dump(
                "The following figure plots the battery capacity, supply and demand curves over the time period of the dataset."
            )
            report.putFig(
                "Battery Capacity vs Supply and Demand Curves",
                self.plotSupplyDemandBatteryCurves(),
            )

        report.saveToPdf("out/report.pdf")
        self.report = None
        command = f"rm ./out/*.png"
        subprocess.run(command, shell=True, text=True, capture_output=True)
        print("PDF report generated at 'out/report.pdf'")

    def printBatteryStats(self) -> None:
        self._printPercent(
            "Consumption covered by battery",
            self.getDischargeVolume() / self.getConsumptionVolume,
        )
        self._printPercent(
            "Percentage of supply put to battery",
            self.getChargeVolume() / self.getSupplyVolume,
        )

    def printKeyStats(self) -> None:
        self._print(
            "Overall consumption (kw/h)",
            self.getConsumptionVolume,
        )
        self._print(
            "Overall production (kw/h)",
            self.getProductionVolume,
        )
        self._print(
            "Overall trading volume (kw/h)",
            self.getTradingVolume,
        )
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
            "Demand covered by trades", self.getTradingVolume / self.getDemandVolume
        )
        self._printPercent(
            "Percentage of supply sold", self.getTradingVolume / self.getSupplyVolume
        )

    def plotSupplyDemandBatteryCurves(self) -> Optional[str]:
        """
        Plots the supply, demand and battery capacity curves over the dataset period.
        """
        batteryCapCurve: pd.Series
        if self.battery is not None:
            batteryCapCurve = self.battery.getCapacityCurve()
        else:
            batteryCapCurve = pd.Series()  # Empty Series if no battery data

        supplyCurve: pd.Series = self.supply.sum(axis=1)
        demandCurve: pd.Series = self.demand.sum(axis=1)

        plt.figure(figsize=(12, 6))

        plt.plot(
            batteryCapCurve.index,
            batteryCapCurve,
            label="Battery Capacity",
            color="blue",
        )
        plt.plot(supplyCurve.index, supplyCurve, label="Supply Curve", color="green")
        plt.plot(demandCurve.index, demandCurve, label="Demand Curve", color="red")
        plt.title("Supply, Demand, and Battery Capacity Curves")
        plt.xlabel("Time Interval")
        plt.ylabel("kw/h")
        plt.legend()
        plt.grid(True)
        return self._show("batteryDemandSupplyCurves")

    def _getHistForRatioValues(
        self, valueVec: pd.Series
    ) -> tuple[pd.Series, NDArray[floating[Any]]]:
        """
        Expects a pd.Series of ratio values (between 0 and 1) and plots a histogram of them,
        putting the values in 100 baskets (corresponding to each percentage point).
        Returns the histogram series to plot.
        """
        # Define bin edges to cover the range from 0 to 1
        bin_edges: NDArray[floating[Any]] = np.linspace(0, 1, N_BINS + 1)

        # Discretize the data into bins
        binned_data = pd.cut(valueVec, bins=bin_edges)

        # Create an IntervalIndex from bin edges
        interval_index = pd.IntervalIndex.from_breaks(bin_edges)

        # Compute histogram counts and reindex to include all bins
        hist: pd.Series = binned_data.value_counts(sort=False).reindex(
            interval_index, fill_value=0
        )
        return hist, bin_edges

    def visualizeSellRatioDistribution(self) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of sold supply.
        """
        sellRatioVec: pd.Series = self.getSellVolumePerMember / self.getSupplyPerMember
        sellRatioVec = sellRatioVec.replace([np.inf], 0)

        hist: pd.Series
        edges: NDArray[floating[Any]]
        hist, edges = self._getHistForRatioValues(sellRatioVec)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black")
        plt.xlabel("Ratio Sold")
        plt.ylabel("Number of Community Members")
        plt.title("Ratio of Sold Supply per Member")
        plt.grid(True)
        return self._show("sellRatioDistr")

    def visualizeBuyRatioDistribution(self) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of purchased demand.
        """
        buyRatioVec: pd.Series = self.getBuyVolumePerMember / self.getDemandPerMember
        buyRatioVec = buyRatioVec.replace([np.inf], 0)

        hist: pd.Series
        edges: NDArray[floating[Any]]
        hist, edges = self._getHistForRatioValues(buyRatioVec)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black")
        plt.xlabel("Ratio Purchased")
        plt.ylabel("Number of Community Members")
        plt.title("Ratio of Purchase Volume / Demand Volume per Member")
        plt.grid(True)
        return self._show("buyRatioDistr")

    def visualizeDischargeRatioDistribution(self) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of purchased demand.
        """
        dischargeRatioVec: pd.Series = (
            self.getDischargeVolumePerMember() / self.getDemandPerMember
        )
        dischargeRatioVec = dischargeRatioVec.replace([np.inf], 0)

        hist: pd.Series
        edges: NDArray[floating[Any]]
        hist, edges = self._getHistForRatioValues(dischargeRatioVec)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black")
        plt.xlabel("Ratio Discharged")
        plt.ylabel("Number of Community Members")
        plt.title("Ratio of Discharge Volume / Demand Volume per Member")
        plt.grid(True)
        return self._show("dischargeRatioDistr")

    def visualizeChargeRatioDistribution(self) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of purchased demand.
        """
        chargeRatioVec: pd.Series = (
            self.getChargeVolumePerMember() / self.getSupplyPerMember
        )
        chargeRatioVec = chargeRatioVec.replace([np.inf], 0)

        hist: pd.Series
        edges: NDArray[floating[Any]]
        hist, edges = self._getHistForRatioValues(chargeRatioVec)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black")
        plt.xlabel("Ratio Charged")
        plt.ylabel("Number of Community Members")
        plt.title("Ratio of Charged Volume / Supply Volume per Member")
        plt.grid(True)
        return self._show("chargeRatioDistribution")

    def _show(self, figName: str) -> Optional[str]:
        name: str = f"./out/{figName}_{self._getUid()}.png"
        match self.report:
            case Report() as r:
                plt.savefig(name)
                return name
            case None:
                plt.show()
                return None

    def _printPercent(self, description: str, num: float) -> None:
        """
        num should be a ratio.
        """
        self._print(description + " (%)", round(num * 100, DECIMAL_POINTS))

    def _print(self, description: str, num: object) -> None:
        formatted_num: str
        if isinstance(num, (float, int)):  # Check if num is a number
            formatted_num = f"{num:.2f}"  # Format to 2 decimal places
        elif isinstance(num, tuple):
            first, second = num
            formatted_num = f"{first}, {second}"
        else:
            formatted_num = str(num)
        match self.report:
            case Report() as r:
                r.dump(f"{description}: {formatted_num}")
            case None:
                print(f"{description}: {formatted_num}")

    @cached_property
    def getTradingVolume(self) -> float:
        """
        Returns the overall trading volume over the timeframe of the dataset.
        """
        return sum(sol.tradingVolume for sol in self.marketSolutions)

    def getDischargeVolume(self) -> float:
        """
        Returns the overall discharging volume over the timeframe of the dataset.
        """
        return sum(sol.dischargeAmount for sol in self.marketSolutions)

    def getChargeVolume(self) -> float:
        """
        Returns the overall charging volume over the timeframe of the dataset.
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
    def getProductionVolume(self) -> float:
        """
        Returns the overall production of all participants over the timeframe of the dataset.
        """
        return self.production.sum().sum()

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

    def _getUid(self) -> int:
        self._uid += 1
        return self._uid
