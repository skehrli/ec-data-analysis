#!/usr/bin/env python3

from ec_dataset import ECDataset
from typing import Optional
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from constants import RETENTION_RATE
from battery import Battery
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import lognorm


def csv_to_df(file_path: str) -> Optional[pd.DataFrame]:
    """
    Reads a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


def getSheet(key: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the requested sheet from the excel dataset with the first
    column dropped (since in the excel sheet, this is the row index).
    Multiply each value by 250 to get w/h, since the entries describe
    the power rate in kw per 15 minute intervals and transpose to have
    the columns be the intervals and rows be the community members.

    Args:
        data (pd.DataFrame): The DataFrame.
        key (str): The sheet to fetch.

    Returns:
        pd.DataFrame: The converted DataFrame with the first column dropped and
        each entry converted to w/h.
    """
    sheet = data[key]
    sheet = sheet.drop(sheet.columns[0], axis=1)
    return sheet / 4


def fit_normal_distribution(df: pd.DataFrame):
    # Dictionary to store results
    distribution_params = {}

    # Iterate over each column
    for column in df.columns:
        # Fit normal distribution to the column data
        mean, std = norm.fit(df[column])

        # Store the results (mean and std deviation) in a dictionary
        distribution_params[column] = {"mean": mean, "std": std}

    return distribution_params


def fit_lognormal_distribution(df: pd.DataFrame, offset: float = 1e-6):
    """
    Fits a log-normal distribution to each column of the DataFrame, adding a small offset
    to handle zero values.

    Args:
        df (pd.DataFrame): A DataFrame where each column contains non-negative data.
        offset (float): A small constant to add to each value to handle zero values.

    Returns:
        dict: A dictionary with column names as keys and another dictionary with
              shape, location, and scale parameters of the log-normal distribution as values.
    """
    # Dictionary to store results
    distribution_params = {}

    # Add offset to handle zero values
    df_offset = df + offset

    # Iterate over each column
    for column in df.columns:
        # Fit log-normal distribution to the column data
        shape, loc, scale = lognorm.fit(df_offset[column], floc=0)

        # Store the results (shape, location, and scale) in a dictionary
        distribution_params[column] = {"shape": shape, "loc": loc, "scale": scale}

    return distribution_params


def plot_distributions(demand_dist, supply_dist):
    for key in demand_dist.keys():
        # Get mean and std for both demand and supply
        mean_demand, std_demand = demand_dist[key]["mean"], demand_dist[key]["std"]
        mean_supply, std_supply = supply_dist[key]["mean"], supply_dist[key]["std"]

        # Create a range of x values (based on the means and stds to cover the distributions)
        x_min = min(mean_demand - 3 * std_demand, mean_supply - 3 * std_supply)
        x_max = max(mean_demand + 3 * std_demand, mean_supply + 3 * std_supply)
        x = np.linspace(x_min, x_max, 1000)

        # Get PDFs for both demand and supply
        demand_pdf = norm.pdf(x, mean_demand, std_demand)
        supply_pdf = norm.pdf(x, mean_supply, std_supply)

        # Plot both curves
        plt.figure(figsize=(8, 6))
        plt.plot(x, demand_pdf, label=f"Demand - {key}", color="blue")
        plt.plot(x, supply_pdf, label=f"Supply - {key}", color="red", linestyle="--")

        # Add labels and title
        plt.title(f"Demand vs Supply Distribution for {key}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_lognormal_comparison(demand_dist: dict, supply_dist: dict):
    for key in demand_dist.keys():
        # Get parameters for both demand and supply distributions
        demand_params = demand_dist[key]
        supply_params = supply_dist[key]

        # Define the range for plotting
        x_min = min(
            demand_params["scale"] - 3 * demand_params["shape"],
            supply_params["scale"] - 3 * supply_params["shape"],
        )
        x_max = max(
            demand_params["scale"] + 3 * demand_params["shape"],
            supply_params["scale"] + 3 * supply_params["shape"],
        )
        x = np.linspace(x_min, x_max, 1000)

        # Compute the PDF for demand and supply distributions
        demand_pdf = lognorm.pdf(
            x,
            s=demand_params["shape"],
            loc=demand_params["loc"],
            scale=demand_params["scale"],
        )
        supply_pdf = lognorm.pdf(
            x,
            s=supply_params["shape"],
            loc=supply_params["loc"],
            scale=supply_params["scale"],
        )

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(x, demand_pdf, label=f"Demand - {key}", color="blue")
        plt.plot(x, supply_pdf, label=f"Supply - {key}", color="red", linestyle="--")

        # Add labels, title, and legend
        plt.title(f"Log-Normal Distribution Comparison for {key}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()


def main() -> None:
    data: pd.DataFrame = pd.read_excel("data/EC_EV_dataset.xlsx", sheet_name=None)
    supply: pd.DataFrame = getSheet("PV", data)
    demand: pd.DataFrame = getSheet("Load", data)

    ecData: ECDataset = ECDataset(supply, demand)
    capacities: np.ndarray = np.linspace(1e2, 1e3, 3)
    ecData.createReport(capacities)

    # Example usage
    # demand_dist = fit_lognormal_distribution(market_demand)
    # supply_dist = fit_lognormal_distribution(market_supply)

    # Example usage:
    # plot_distributions(demand_dist, supply_dist)
    # plot_lognormal_comparison(demand_dist, supply_dist)


if __name__ == "__main__":
    main()
