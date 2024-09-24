#!/usr/bin/env python3

"""
ec_dataset.py

This module contains the ECDataset class, which is used to manage and manipulate
datasets for energy communities.

TODO add this documentation
"""

from .battery import Battery
from .constants import DECIMAL_POINTS, EPS, Constants, TEXTS, OUT
from .report import Report
from .market_solution import MarketSolution
from .plot_utils import PlotUtils
import pandas as pd
import matplotlib.pyplot as plt
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

    # duration of one timestep as a fraction/multiple of an hour
    timestepDuration: float

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
        timestepDuration: float,
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
        self.timestepDuration = timestepDuration
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

    def changeBattery(
        self, newBattery: Battery, earlyTermination: bool = False
    ) -> bool:
        """
        Returns whether the battery capacity is sufficient to fit all of the unsold supply.
        If earlyTermination is set, the method returns False upon the first occurence of
        a timestep where the battery is not sufficiently large, without computing the
        following timesteps.
        """
        newBattery.reset()
        self.battery = newBattery
        sufficient: bool = True
        for t in range(self.numTimesteps):
            if not self.marketSolutions[t].computeWithBattery(newBattery):
                if earlyTermination:
                    return False
                sufficient = False
        return sufficient

    def createReport(self) -> None:
        report: Report = Report("EC Report")
        self.report = report
        report.dumpFile(f"{TEXTS}/introduction.md")

        report.addSection("Key Statistics Without Battery")
        report.dumpFile(f"{TEXTS}/keyStats.md")
        self.printKeyStats()

        report.addHeading("Energy Consumption/Production by Source", 2)
        report.dumpFile(f"{TEXTS}/energyBreakdown.md")
        report.putFigs(
            self.visualizeEnergyConsumptionBreakdown(),
            self.visualizeEnergyProductionBreakdown(),
        )

        report.addHeading("Sell Ratio", 2)
        report.dumpFile(f"{TEXTS}/sellRatio.md")
        report.putFig("Sell Ratio", self.visualizeSellRatioDistribution())

        report.addHeading("Buy Ratio", 2)
        report.dumpFile(f"{TEXTS}/buyRatio.md")
        report.putFig("Buy Ratio Distribution", self.visualizeBuyRatioDistribution())

        report.addSection("Statistics With Battery")
        report.dumpFile(f"{TEXTS}/statsWithBattery.md")

        self._findOptimalBattery()
        self.printBatteryStats()

        report.addHeading("Energy Consumption/Production By Source", 2)
        report.dumpFile(f"{TEXTS}/energyBreakdownWithBattery.md")
        report.putFigs(
            self.visualizeEnergyConsumptionBreakdown(),
            self.visualizeEnergyProductionBreakdown(),
        )

        report.addHeading("Ratio of energy obtained from battery of overall demand", 2)
        report.dumpFile(f"{TEXTS}/dischargeRatio.md")
        report.putFig(
            "Battery Discharge Ratio", self.visualizeDischargeRatioDistribution()
        )

        report.addHeading("Ratio of energy sold to battery of overall supply", 2)
        report.dumpFile(f"{TEXTS}/chargeRatio.md")
        report.putFig("Battery Charge Ratio", self.visualizeChargeRatioDistribution())

        report.addHeading("Supply, Demand and Battery Capacity Curve", 2)
        report.dumpFile(f"{TEXTS}/supplyDemandCapacityCurves.md")
        report.putFig(
            "Battery Capacity vs Supply and Demand Curves",
            self.plotSupplyDemandBatteryCurves(),
        )

        report.saveToPdf(f"{OUT}/report.pdf")
        self.report = None
        command = f"rm ./{OUT}/*.png"
        subprocess.run(command, shell=True, text=True, capture_output=True)
        print(f"PDF report generated at '{OUT}/report.pdf'")

    def printBatteryStats(self) -> None:
        match self.battery:
            case Battery():
                self._print(
                    "Required Battery Capacity for Full Communal Self-Consumption (kw/h)",
                    self.battery.capacity,
                )
                self._printPercent(
                    "Consumption covered by battery",
                    self.getDischargeVolume() / self.getConsumptionVolume,
                )
                self._printPercent(
                    "Percentage of supply put to battery",
                    self.getChargeVolume() / self.getSupplyVolume,
                )
            case None:
                raise ValueError("printBatteryStats() called, but no Battery set.")

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
            self.compareProductionWithConsumption,
        )
        # self._printPercent(
        #     "Consumption covered by own production",
        #     (self.getConsumptionVolume - self.getDemandVolume)
        #     / self.getConsumptionVolume,
        # )
        # self._printPercent(
        #     "Consumption covered by trades",
        #     self.getTradingVolume / self.getConsumptionVolume,
        # )
        self._printPercent(
            "Demand covered by trades", self.getTradingVolume / self.getDemandVolume
        )
        self._printPercent(
            "Percentage of supply sold", self.getTradingVolume / self.getSupplyVolume
        )

    def visualizeEnergyConsumptionBreakdown(self) -> Optional[str]:
        """
        Plots where the consumed energy comes from (self-production, market, battery, grid).
        """
        # Retrieve energy breakdown for consumption
        consumption_values = [
            self.getSelfConsumptionVolume,  # Energy consumed from self-production
            self.getTradingVolume,  # Energy consumed from the market
            self.getDischargeVolume(),  # Energy consumed from battery discharge
            self.getGridPurchaseVolume(),  # Energy consumed from the grid
        ]

        PlotUtils.createEnergyBreakdownDonutChart(
            "Energy Consumption by Source", consumption_values
        )
        return self._show("EnergyBreakdownConsumption")

    def visualizeEnergyProductionBreakdown(self) -> Optional[str]:
        """
        Plots where the produced energy goes to (self-consumption, market, battery, grid).
        """
        # Retrieve energy breakdown for production
        production_values = [
            self.getSelfConsumptionVolume,  # Energy used in self-consumption
            self.getTradingVolume,  # Energy sold to the market
            self.getChargeVolume(),  # Energy stored in the battery
            self.getGridFeedInVolume(),  # Energy fed back into the grid
        ]

        PlotUtils.createEnergyBreakdownDonutChart(
            "Energy Production by Destination", production_values
        )
        return self._show("EnergyBreakdownProduction")

    def plotSupplyDemandBatteryCurves(self) -> Optional[str]:
        """
        Plots the supply, demand and battery capacity curves over the dataset period.
        """
        if self.battery is not None:
            batteryCapCurve: pd.Series = self.battery.getCapacityCurve()
            supplyCurve: pd.Series = self.supply.sum(axis=1)
            demandCurve: pd.Series = self.demand.sum(axis=1)

            palette = Constants.getColorPalette(6)
            plt.figure(figsize=(12, 6))

            plt.plot(
                batteryCapCurve.index,
                batteryCapCurve,
                label="Battery Capacity",
                color=palette[0],
            )
            plt.plot(
                supplyCurve.index,
                supplyCurve,
                label="Supply Curve",
                color=palette[1],
            )
            plt.plot(
                demandCurve.index,
                demandCurve,
                label="Demand Curve",
                color=palette[2],
            )
            plt.axhline(
                y=self.battery.capacity,
                color=palette[3],
                linestyle="--",
                label="Max Capacity (Dashed)",
            )
            plt.axhline(
                y=self.battery.maxAllowedCharge,
                color=palette[4],
                linestyle="--",
                label="Max Allowed Charge (Dashed)",
            )
            plt.axhline(
                y=self.battery.minAllowedCharge,
                color=palette[5],
                linestyle="--",
                label="Min Allowed Charge (Dashed)",
            )
            plt.title("Supply, Demand, and Battery Capacity Curves")
            plt.xlabel("Time Interval")
            plt.ylabel("kw/h")
            plt.legend()
            plt.grid(True)
            return self._show("batteryDemandSupplyCurves")
        else:
            raise ValueError("Cannot plot battery curve - no battery assigned.")

    def visualizeSellRatioDistribution(self) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of sold supply.
        """
        sellRatioVec: pd.Series = self.getSellVolumePerMember / self.getSupplyPerMember
        sellRatioVec = sellRatioVec.replace([np.inf], 0)

        hist: pd.Series
        edges: NDArray[floating[Any]]
        hist, edges = PlotUtils.getHistForRatioValues(sellRatioVec)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(
            edges[:-1],
            hist,
            width=np.diff(edges),
            color=Constants.getColorPalette(1),
            edgecolor="black",
        )
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
        hist, edges = PlotUtils.getHistForRatioValues(buyRatioVec)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(
            edges[:-1],
            hist,
            width=np.diff(edges),
            color=Constants.getColorPalette(1),
            edgecolor="black",
        )
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
        hist, edges = PlotUtils.getHistForRatioValues(dischargeRatioVec)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(
            edges[:-1],
            hist,
            width=np.diff(edges),
            color=Constants.getColorPalette(1),
            edgecolor="black",
        )
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
        hist, edges = PlotUtils.getHistForRatioValues(chargeRatioVec)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(
            edges[:-1],
            hist,
            width=np.diff(edges),
            color=Constants.getColorPalette(1),
            edgecolor="black",
        )
        plt.xlabel("Ratio Charged")
        plt.ylabel("Number of Community Members")
        plt.title("Ratio of Charged Volume / Supply Volume per Member")
        plt.grid(True)
        return self._show("chargeRatioDistribution")

    def _show(self, figName: str) -> Optional[str]:
        name: str = f"./{OUT}/{figName}_{self._getUid()}.png"
        match self.report:
            case Report():
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

    def getGridFeedInVolume(self) -> float:
        """
        Returns the overall energy fed into the grid over the timeframe of the dataset.
        """
        return max(
            0,
            self.getProductionVolume
            - self.getSelfConsumptionVolume
            - self.getTradingVolume
            - self.getChargeVolume(),
        )

    def getGridPurchaseVolume(self) -> float:
        """
        Returns the overall energy purchased from the grid over the timeframe of the dataset.
        """
        return max(
            0,
            self.getConsumptionVolume
            - self.getSelfConsumptionVolume
            - self.getTradingVolume
            - self.getDischargeVolume(),
        )

    @cached_property
    def compareProductionWithConsumption(self) -> tuple[int, int]:
        return (self.supply > 0).sum().sum(), (self.demand > 0).sum().sum()

    @cached_property
    def getSelfConsumptionVolume(self) -> float:
        """
        Returns the volume of self-consumed energy over the timeframe of the dataset.
        """
        selfConsumption: pd.DataFrame = self.production - self.supply
        return selfConsumption.sum().sum()

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

    def _findOptimalBattery(self) -> None:
        """
        Binary searches for optimal battery capacity.
        """
        battery: Battery
        l: float = 1.0
        r: float
        while True:
            battery = Battery(l, self.timestepDuration)
            if self.changeBattery(battery, True):
                break
            else:
                l *= 2
        r = l
        l /= 2
        cap: float
        while abs(r - l) > EPS:
            cap = (r + l) / 2
            battery = Battery(cap, self.timestepDuration)
            if self.changeBattery(battery, True):
                r = cap
            else:
                l = cap
        self.changeBattery(battery)

    def _getUid(self) -> int:
        self._uid += 1
        return self._uid
