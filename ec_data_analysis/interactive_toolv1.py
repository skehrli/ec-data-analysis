#!/usr/bin/env python3

"""
this is similar to ec_dataset.py, but adding battery self-consumption to init

This module similar to the ECDataset class; used to simulate P2P local energy market (LEM) with max-flow algorithm
(with or without fairness constraints).
Might take away parts about making a PDF report.!!!!
"""

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


def aggregate_profiles(n_agg, load_profiles):
    # n_agg is the number of loads to aggregate (e.g. 6 for 6-apt building)
    agg_df = pd.DataFrame()
    for i in range(0, len(load_profiles.columns) - 1, n_agg):
        new_col_name = f'agg_profile{i // n_agg + 1}'
        agg_df[new_col_name] = load_profiles.iloc[:, i:i + n_agg-1].sum(axis=1)

    return agg_df

################################################

def apply_load_shifting(df_load, df_generation, dsm_percentage):
    """
    Attempts to reduce peak loads by shifting loads for a subset of users while increasing PV consumption,
    and ensuring that loads are not shifted outside 8 AM - 10 PM.

    Parameters:
    - df_load: DF with hourly load profiles
    - df_generation: DF with hourly PV generation profiles 
    - dsm_percentage: Percentage of users capable of shifting load (DSM)

    Returns:
    - df_load_shifted: New df with adjusted load profiles
    """

    df_shifted = df_load.copy()
    users = df_load.columns
    shiftable_users = np.random.choice(users, size=int(len(users) * dsm_percentage/100), replace=False)

    # Dishwasher and washing machine shifting amounts (daily and every 3 days)
    shift_daily = 0.64  # kWh
    shift_3day = 0.5  # kWh

    # convert timestamps to date for grouping
    df_load["date"] = df_load.index.date
    df_generation["date"] = df_generation.index.date
    grouped_load = df_load.groupby("date")
    grouped_generation = df_generation.groupby("date")

    for day, data in grouped_load:

        # find peak hours of each day in allowed time range
        N_peaks = 3  # number of peaks
        total_demand = data.drop(columns=["date"], errors="ignore").sum(axis=1)

        valid_hours = total_demand.between_time("08:00", "22:00")

        peak_indices, _ = find_peaks(valid_hours, prominence=0.2, distance=N_peaks)
        peak_hours = valid_hours.iloc[peak_indices].nlargest(3).index if len(
            peak_indices) > 0 else valid_hours.nlargest(3).index
        print("peak hours: ", peak_hours)

        # find valley hours in allowed time range
        valley_hours = valid_hours.nsmallest(N_peaks).index  # lowest 3 hours
        print("valley hours: ", valley_hours)

        # Get PV generation for the same day
        pv_data = grouped_generation.get_group(day).drop(columns=["date"], errors="ignore")
        pv_generation = pv_data.sum(axis=1)
        # all 3 highest pv hours are probably from same "pv peak"
        high_pv_hours = pv_generation.nlargest(3).index  # assuming these are in the valid range of hours.
        print("high pv hours:", high_pv_hours)

        # shift daily load (dishwasher)
        if len(peak_hours) > 0:
            peak_hour = peak_hours[0]  # take the highest peak
            for user in shiftable_users:
                if df_shifted.loc[peak_hour, user] >= shift_daily:  # user has enough load to shift
                    # try to shift to high PV generation hours
                    shift_target = np.random.choice(high_pv_hours) if not high_pv_hours.empty else np.random.choice(
                        valley_hours)
                    # shift the load
                    df_shifted.loc[peak_hour, user] -= shift_daily
                    df_shifted.loc[shift_target, user] += shift_daily

        # shift every 3 days (washing machine)
        if day.day % 3 == 0 and len(peak_hours) > 0:
            peak_hour = peak_hours[1] if len(peak_hours) > 1 else peak_hours[0]  # try to use 2nd highest peak
            for user in shiftable_users:
                if df_shifted.loc[peak_hour, user] >= shift_3day:
                    shift_target = np.random.choice(high_pv_hours) if not high_pv_hours.empty else np.random.choice(
                        valley_hours)
                    df_shifted.loc[peak_hour, user] -= shift_3day
                    df_shifted.loc[shift_target, user] += shift_3day

    df_shifted.drop(columns=["date"], inplace=True, errors='ignore')

    return df_shifted
################################################


def consider_price(
        excess_prod, unsatisfied_load, soldPerTS, boughtPerTS, PGB_1=12, PGB_2=15, PGS=5, PP2P=10) -> pd.DataFrame:

    # excess_prod and unsatisfied_load already without timestamp

    # get price grid buy (PGB) values for different times.
    ### for cired-type lecs with timestamp ###
    # PGB = self.unsatisfied_load.index.to_series().apply(lambda ts: PGB_2 if 6 <= ts.hour <= 22 else PGB_1)
    # PGB = PGB.reset_index(drop=True)
    ######
    ### for ec-ev data starts at 00:00 ###
    size = 96
    PGB = pd.Series(15, index=range(size))
    PGB.loc[0:23] = 12  # 00:00 to 06:00
    PGB.loc[87:95] = 12  # 22:00 to 24:00
    # print("PGB: ", PGB)

    # soldPerTS = self.getSellVolumePerMemberPerTS  # sold in P2P market
    # boughtPerTS = self.getBuyVolumePerMemberPerTS  # bought from P2P market

    excess_prod.columns = soldPerTS.columns  # might not need this??
    unsatisfied_load.columns = boughtPerTS.columns

    soldGridPerTS = excess_prod - soldPerTS  # sold to Grid
    boughtGridPerTS = unsatisfied_load - boughtPerTS  # bought from Grid

    # Transaction Value: BUYING from P2P Market
    TV_buyP2P = boughtPerTS * PP2P

    # Transaction Value: BUYING from Grid
    TV_buyGrid = boughtGridPerTS.multiply(PGB, axis=0)

    # Transaction Value: SELLING to P2P Market
    TV_sellP2P = soldPerTS * PP2P

    # Transaction Value: SELLING TO Grid
    TV_sellGrid = soldGridPerTS * PGS

    TV_PerMemberPerTS = TV_sellP2P + TV_sellGrid - TV_buyP2P - TV_buyGrid

    return TV_PerMemberPerTS



class simulate_LEC:
    # The 4 core DataFrames
    production: pd.DataFrame
    consumption: pd.DataFrame
    excess_prod: pd.DataFrame
    unsatisfied_load: pd.DataFrame

    # Dimensions of the above DataFrames: rows/columns
    numParticipants: int
    numTimesteps: int

    # duration of one timestep as a fraction/multiple of an hour
    timestepDuration: float

    # solutions to the flow problem per time interval
    marketSolutions: List[MarketSolution] = []

    # used to compile a pdf report
    report: Optional[Report]
    _uid: int

    def __init__(
        self,
        production: pd.DataFrame,
        consumption: pd.DataFrame,
        lec_flag: bool,  # always FAIR for interactive tool case.
        timestepDuration: float,
        batt_vec: pd.Series,
        battery: Optional[Battery] = None,
    ) -> None:
        """
        Initialize the data class with production and consumption DataFrames.

        :param production: DataFrame containing production data. production[i][j] is the production in w/h of member j in time interval i.
        :param consumption: DataFrame containing consumption data. consumption[i][j] is the usage in w/h of member j in time interval i.
        :param batt_vec: as long as the number of LEC members, 0 for pure consumers, 1 for prosumers (PV-able)

        :ivar production: DataFrame with production[i][j] being the production in w/h of member j in time interval i.
        :ivar consumption: DataFrame with consumption[i][j] being the usage in w/h of member j in time interval i.
        :ivar excess_prod: DataFrame with sell[i][j] being the amount member j is selling in interval i.
        :ivar unsatisfied_load: DataFrame with buy[i][j] being the amount member j is buying in interval i.
        """
        assert (
            production.shape == consumption.shape
        ), f"production and consumption have unequal shapes {production.shape} and {consumption.shape}"
        self.lec_flag = lec_flag  # this will change load-shiting function and then whether there is trading.
        self.timestepDuration = timestepDuration
        self.numTimesteps, self.numParticipants = production.shape
        production.columns = range(self.numParticipants)
        consumption.columns = range(self.numParticipants)
        self.production = production
        self.consumption = consumption

        # first: PV SELF-CONSUMPTION
        self.excess_prod = (self.production - self.consumption).clip(lower=0)
        self.unsatisfied_load = (self.consumption - self.production).clip(lower=0)

        # second: optional BATTERY SELF-CONSUMPTION
        self.battery =  battery
        self.batt_vec = batt_vec
        batt_capacity = 20  # will this be an input? 20kw for 6-apt building, 10kwh for single households.!!!
        LOWER_THRESHOLD = 0.15  # maybe move these constants
        HIGHER_THRESHOLD = 0.95
        EFFICIENCY = 0.9
        C_RATE = 0.2
        if self.battery is not None:
            soc = np.full(len(self.batt_vec), LOWER_THRESHOLD * batt_capacity)  # initialize SoC
            maxAllowedCharge = HIGHER_THRESHOLD * batt_capacity
            minAllowedCharge = LOWER_THRESHOLD * batt_capacity
            cRateLimit = C_RATE * self.timestepDuration * batt_capacity

            for t in range(self.numTimesteps):
                for user in range(self.numParticipants):
                    if batt_vec[user] == 1:  # member has battery
                        if self.unsatisfied_load.iloc[t, user] > 0:  # load exceeds gen, attempt to DISCHARGE batt
                            amount = self.unsatisfied_load.iloc[t, user]
                            maxDischargeAmount = max(0.0, soc[user] - minAllowedCharge)
                            discharge = min(amount, cRateLimit, maxDischargeAmount)
                            soc[user] -= discharge / EFFICIENCY
                            self.unsatisfied_load.iloc[t, user] -= discharge  # make a copy and save in different df????
                            # excess generation stays the same
                        elif self.excess_prod.iloc[t, user] > 0:  # surplus gen, attempt to CHARGE batt
                            amount = self.excess_prod.iloc[t, user]
                            maxChargeAmount = max(0.0, maxAllowedCharge - soc[user])
                            charge = min(amount, cRateLimit, maxChargeAmount)
                            soc[user] += charge * EFFICIENCY
                            self.excess_prod.iloc[t, user] -= charge  # make a copy and save in different df?????
                            # unsatisfied load stays the same

        # output from 2nd step is the final excess_prod and unsatisfied_load

        # third: TRADING IN LEC
        self.production = self.production.reset_index(drop=True)  # shouldn't need timestamp anymore ... can eliminate??
        self.consumption = self.consumption.reset_index(drop=True)  # same, can eliminate??
        self.report = None
        if self.lec_flag:   # LEFT OFF HERE!!!!! ... what to do if ELSE? i.e. NO LEC.
            df: pd.DataFrame = self.excess_prod - self.unsatisfied_load  # df = prod-cons > 0 when selling
            for t in range(self.numTimesteps):
                self.marketSolutions.append(MarketSolution(df.iloc[t]))

        self._uid = 0

        # outputs needed: quantity sold in p2p/ext-grid, quantity sold in p2p/ext-grid -> with that, economic benefit.


    def createReport(self) -> None:
        report: Report = Report("EC Report")
        self.report = report
        report.dumpFile("introduction.md")

        report.addSection("Key Statistics Without Battery")
        report.dumpFile("keyStats.md")
        self.printKeyStats()

        report.addHeading("Energy Consumption/Production by Source", 2)
        report.dumpFile("energyBreakdown.md")
        report.putFigs(
            self.visualizeEnergyConsumptionBreakdown(),
            self.visualizeEnergyProductionBreakdown(),
        )

        report.addHeading("Sell Ratio", 2)
        report.dumpFile("sellRatio.md")
        report.putFig("Sell Ratio", self.visualizeSellRatioDistribution(True))

        report.addHeading("Sell Ratio without Per-Timestep Fairness", 2)
        report.dumpFile("unfairSellRatio.md")
        report.putFig(
            "Sell Ratio Without Fairness",
            self.visualizeSellRatioDistribution(False)
        )

        report.addHeading("Buy Ratio", 2)
        report.dumpFile("buyRatio.md")
        report.putFig("Buy Ratio Distribution", self.visualizeBuyRatioDistribution(True))

        report.addHeading("Buy Ratio without Per-Timestep Fairness", 2)
        report.dumpFile("unfairBuyRatio.md")
        report.putFig(
            "Buy Ratio Distribution Without Fairness",
            self.visualizeBuyRatioDistribution(False),
        )

        # fix bug here in case there is no "out" folder
        report.saveToPdf(f"{OUT}/report.pdf")
        self.report = None
        command = f"rm ./{OUT}/*.png"
        subprocess.run(command, shell=True, text=True, capture_output=True)
        print(f"PDF report generated at '{OUT}/report.pdf'")


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
        self._printPercent(
            "Percentage of load not satisfied by self-production bought from P2P market",
            self.getTradingVolume / self.getTotalUnsatLoad
        )
        self._printPercent(
            "Percentage of excess generation sold to P2P market", self.getTradingVolume / self.getTotalExcessProd
        )

    def visualizeEnergyConsumptionBreakdown(self) -> Optional[str]:
        # Plots where the consumed energy comes from (self-production, market, grid).
        # Retrieve energy breakdown for consumption
        consumption_values = [
            self.getSelfConsumptionVolumeLoad,  # Energy consumed from self-production
            self.getTradingVolume,  # Energy consumed from the market
            self.getGridPurchaseVolume(),  # Energy consumed from the grid
        ]
        print("energy consumption by source: ", consumption_values)

        PlotUtils.createEnergyBreakdownDonutChart(
            "Energy Consumption by Source", consumption_values
        )
        return self._show("EnergyBreakdownConsumption")

    def visualizeEnergyProductionBreakdown(self) -> Optional[str]:
        """
        Plots where the produced energy goes to (self-consumption, market, grid).
        """
        # Retrieve energy breakdown for production
        production_values = [
            self.getSelfConsumptionVolumeProd,  # Energy used in self-consumption
            self.getTradingVolume,  # Energy sold to the market
            self.getGridFeedInVolume(),  # Energy fed back into the grid
        ]

        print("energy production by destination: ", production_values)

        PlotUtils.createEnergyBreakdownDonutChart(
            "Energy Production by Destination", production_values
        )
        return self._show("EnergyBreakdownProduction")

    def plotSellBuyBatteryCurves(self) -> Optional[str]:
        """
        Plots the excess_gen, unsatisfied_load
        """
        excessGenCurve: pd.Series = self.excess_prod.sum(axis=1)
        unsatisfiedLoadCurve: pd.Series = self.unsatisfied_load.sum(axis=1)
        palette = Constants.getColorPalette(6)
        plt.figure(figsize=(12, 6))
        plt.plot(
            excessGenCurve.index,
            excessGenCurve,
            label="sell Curve",
            color=palette[1],
        )
        plt.plot(
            unsatisfiedLoadCurve.index,
            unsatisfiedLoadCurve,
            label="Buy Curve",
            color=palette[2],
        )
        plt.title("Excess generation and unsatisfied load curves")
        plt.xlabel("Time Interval")
        plt.ylabel("kw/h")
        plt.legend()
        plt.grid(True)
        return self._show("BuySellCurves")



    def visualizeSellRatioDistribution(self, fair: bool) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of sold excess generation.
        """
        sellRatioVec: pd.Series = (self.getSellVolumePerMember if fair
                                   else self.getUnfairSellVolumePerMember) / self.getAvailableSellPerMember
        # sellRatioVec: pd.Series = self.getSellVolumePerMember / self.getAvailableSellPerMember
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
        plt.title("Ratio of Sold Sell per Member")
        plt.grid(True)
        return self._show("sellRatioDistr")

    def visualizeBothSellRatioDistribution(self) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of sold excess generation.
        """
        sellRatioVec_fair: pd.Series = self.getSellVolumePerMember / self.getAvailableSellPerMember
        sellRatioVec_unfair: pd.Series = self.getUnfairSellVolumePerMember / self.getAvailableSellPerMember
        # sellRatioVec: pd.Series = self.getSellVolumePerMember / self.getAvailableSellPerMember
        sellRatioVec_fair = sellRatioVec_fair.replace([np.inf], 0)
        sellRatioVec_unfair = sellRatioVec_unfair.replace([np.inf], 0)

        hist: pd.Series
        edges: NDArray[floating[Any]]
        hist, edges = PlotUtils.getHistForRatioValues(sellRatioVec_fair)
        hist1, edges1 = PlotUtils.getHistForRatioValues(sellRatioVec_unfair)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(
            edges[:-1],
            hist,
            width=np.diff(edges),
            color="#1f77b4",
            edgecolor="black",
            alpha=0.7,
            label="With fairness criteria"
        )
        plt.bar(
            edges1[:-1],
            hist1,
            width=np.diff(edges1),
            color="#ff7f0e",
            edgecolor="black",
            alpha=0.7,
            label="Without fairness criteria"
        )
        plt.xlabel("Ratio Sold", fontsize=18)
        plt.ylabel("Number of Community Members", fontsize=18)
        plt.title("Ratio of Energy Sold in P2P Market", fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=18)
        return self._show("sellRatioDistr")

    def visualizeBuyRatioDistribution(self, fair: bool) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of purchased Buy.
        """
        buyRatioVec: pd.Series = (self.getBuyVolumePerMember if fair
                                  else self.getUnfairBuyVolumePerMember) / self.getUnsatLoadPerMember
        # buyRatioVec: pd.Series = self.getBuyVolumePerMember / self.getUnsatLoadPerMember
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
        plt.title("Ratio of Purchase Volume / Buy Volume per Member")
        plt.grid(True)
        return self._show("buyRatioDistr")

    def visualizeBothBuyRatioDistribution(self) -> Optional[str]:
        """
        Make a bar chart visualizing the distribution of the ratio of purchased Buy.
        """
        buyRatioVec_fair: pd.Series = self.getBuyVolumePerMember / self.getUnsatLoadPerMember
        buyRatioVec_unfair: pd.Series = self.getUnfairBuyVolumePerMember / self.getUnsatLoadPerMember
        buyRatioVec_fair = buyRatioVec_fair.replace([np.inf], 0)
        buyRatioVec_unfair = buyRatioVec_unfair.replace([np.inf], 0)

        hist: pd.Series
        edges: NDArray[floating[Any]]
        hist, edges = PlotUtils.getHistForRatioValues(buyRatioVec_fair)
        hist1, edges1 = PlotUtils.getHistForRatioValues(buyRatioVec_unfair)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(
            edges[:-1],
            hist,
            width=np.diff(edges),
            color="#1f77b4",
            edgecolor="black",
            alpha=0.7,
            label="With fairness criteria"
        )
        plt.bar(
            edges1[:-1],
            hist1,
            width=np.diff(edges1),
            color="#ff7f0e",
            edgecolor="black",
            alpha=0.7,
            label="Without fairness criteria"
        )
        plt.xlabel("Ratio Purchased", fontsize=18)
        plt.ylabel("Number of Community Members", fontsize=18)
        plt.title("Ratio of Energy Purchased from P2P Market", fontsize=20)
        plt.grid(True)
        plt.legend(fontsize=18)
        return self._show("buyRatioDistr")

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
            - self.getSelfConsumptionVolumeProd
            - self.getTradingVolume
        )

    def getGridPurchaseVolume(self) -> float:
        """
        Returns the overall energy purchased from the grid over the timeframe of the dataset.
        """
        return max(
            0,
            self.getConsumptionVolume
            - self.getSelfConsumptionVolumeLoad
            - self.getTradingVolume
        )

    @cached_property
    def compareProductionWithConsumption(self) -> tuple[int, int]:
        return (self.excess_prod > 0).sum().sum(), (self.unsatisfied_load > 0).sum().sum()

    @cached_property
    def getSelfConsumptionVolumeProd(self) -> float:
        """
        Returns the volume of self-consumed produced energy over the timeframe of the dataset.
        """
        selfConsumption: pd.DataFrame = self.production - self.excess_prod  # to consider possible batt (ind)!
        return selfConsumption.sum().sum()

    @cached_property
    def getSelfConsumptionVolumeLoad(self) -> float:
        """
        Returns the volume of self-consumed energy demand over the timeframe of the dataset.
        """
        selfConsumption: pd.DataFrame = self.consumption - self.unsatisfied_load  # to consider possible batt (ind)!
        return selfConsumption.sum().sum()

    @cached_property
    def getTradingVolume(self) -> float:
        """
        Returns the overall trading volume over the timeframe of the dataset.
        """
        return sum(sol.tradingVolume for sol in self.marketSolutions)

    @cached_property
    def getTotalUnsatLoad(self) -> float:
        """
        Returns the overall Buy on the market over the timeframe of the dataset.
        """
        return self.unsatisfied_load.sum().sum()

    @cached_property
    def getTotalExcessProd(self) -> float:
        """
        Returns the overall excess production on the market over the timeframe of the dataset.
        """
        return self.excess_prod.sum().sum()

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
    def getUnsatLoadPerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall Buy.
        """
        return self.unsatisfied_load.sum()

    @cached_property
    def getAvailableSellPerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall Sell.
        """
        return self.excess_prod.sum()

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
    def getUnfairSellVolumePerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall sell volume in a market case without fairness.
        """
        return pd.Series(
            {
                i: sum(sol.getUnfairQtySoldForMember(i) for sol in self.marketSolutions)
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
    def getUnfairBuyVolumePerMember(self) -> pd.Series:
        """
        Returns a map from participant to its overall buy volume in a market case without fairness.
        """
        return pd.Series(
            {
                i: sum(
                    sol.getUnfairQtyPurchasedForMember(i) for sol in self.marketSolutions
                )
                for i in range(self.numParticipants)
            }
        )

    @cached_property
    def getSellVolumePerMemberPerTS(self) -> pd.DataFrame:
        # each participant is a row and each market solution is a column ... NOT ANYMORE.
        # each market solution is a row and each participant is a column.
        data = {
            i: [sol.getQtySoldForMember(i) for sol in self.marketSolutions]
            for i in range(self.numParticipants)
        }

        # df = pd.DataFrame(data).transpose()
        df = pd.DataFrame(data)
        # df.columns = [f"MarketSolution_{j}" for j in range(len(self.marketSolutions))]
        return df

    @cached_property
    def getUnfairSellVolumePerMemberPerTS(self) -> pd.DataFrame:
        # each participant is a row and each market solution is a column .... NOT ANYMORE.
        data = {
            i: [sol.getUnfairQtySoldForMember(i) for sol in self.marketSolutions]
            for i in range(self.numParticipants)
        }

        # df = pd.DataFrame(data).transpose()
        df = pd.DataFrame(data)
        # df.columns = [f"MarketSolution_{j}" for j in range(len(self.marketSolutions))]
        return df

    @cached_property
    def getBuyVolumePerMemberPerTS(self) -> pd.DataFrame:
        # each participant is a row and each market solution is a column ... NOT ANYMORE.
        data = {
            i: [sol.getQtyPurchasedForMember(i) for sol in self.marketSolutions]
            for i in range(self.numParticipants)
        }

        # df = pd.DataFrame(data).transpose()
        df = pd.DataFrame(data)
        # df.columns = [f"MarketSolution_{j}" for j in range(len(self.marketSolutions))]
        return df

    @cached_property
    def getUnfairBuyVolumePerMemberPerTS(self) -> pd.DataFrame:
        # each participant is a row and each market solution is a column ... NOT ANYMORE.
        data = {
            i: [sol.getUnfairQtyPurchasedForMember(i) for sol in self.marketSolutions]
            for i in range(self.numParticipants)
        }

        # df = pd.DataFrame(data).transpose()
        df = pd.DataFrame(data)
        # df.columns = [f"MarketSolution_{j}" for j in range(len(self.marketSolutions))]
        return df

    def _getUid(self) -> int:
        self._uid += 1
        return self._uid
