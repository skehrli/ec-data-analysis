#!/usr/bin/env python3

from ec_data_analysis import ECDataset
import pandas as pd


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

def main() -> None:
    data: pd.DataFrame = pd.read_excel("data/EC_EV_dataset.xlsx", sheet_name=None)
    production: pd.DataFrame = getSheet("PV", data)
    consumption: pd.DataFrame = getSheet("Load", data)

    ecData: ECDataset = ECDataset(production, consumption, 0.25)
    ecData.createReport()

if __name__ == "__main__":
    main()
