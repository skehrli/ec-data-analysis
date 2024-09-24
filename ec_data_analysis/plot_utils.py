#!/usr/bin/env python3

import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from typing import List, Any
import pandas as pd
import numpy as np
from numpy import floating
from numpy.typing import NDArray
from .constants import Constants
from .constants import N_BINS


class PlotUtils:
    @staticmethod
    def createEnergyBreakdownDonutChart(title: str, values: List[float]) -> None:
        # Normalize the values so that the total height equals 1
        total = sum(values)
        ratios = [value / total for value in values]  # Get ratios

        # Define labels for each component
        labels = ["Self-Consumption", "Market", "Battery", "Grid"]

        # Define colors for each component
        colors = Constants.getColorPalette(len(labels))
        # Create the donut chart
        fig, ax = plt.subplots()

        # Create a pie chart and remove the center to create a donut chart
        ax.pie(
            ratios,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.4},
        )
        # Add a circle in the middle to create the donut hole
        center_circle = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(center_circle)

        ax.set_aspect("equal")
        plt.title(title)
        plt.tight_layout()

    @staticmethod
    def getHistForRatioValues(
        valueVec: pd.Series,
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
