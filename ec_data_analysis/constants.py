#!/usr/bin/env python3

from typing import TypeAlias
import seaborn as sns

NetworkAlloc: TypeAlias = dict[str, dict[str, float]]

# flow network
SOURCE: str = "s"
BATTERY: str = "b"
TARGET: str = "t"
UNBOUNDED: float = float("inf")

# numerical error margin
EPS: float = 1e-7

# decimal points to round to
DECIMAL_POINTS: int = 2

# number of bins for bar charts
N_BINS: int = 100

# battery retention rate per hour
RETENTION_RATE: float = 0.999
# conversion loss for every transaction with battery
CONVERSION_LOSS: float = 0.05
# (dis)charging rate - takes at least 1/C_RATE hours to (dis)charge battery
C_RATE: float = 0.5
# minimum allowed charging level
DISCHARGE_THRESHOLD = 0.15
# maximum allowed charging level
CHARGE_THRESHOLD = 1 - DISCHARGE_THRESHOLD

# seaborn color palette used for plots
COLOR_PALETTE: str = "deep"

# subdirectories
OUT: str = "out"
TEXTS: str = "report_texts"


class Constants:
    @staticmethod
    def getColorPalette(numColors: int):
        return sns.color_palette(COLOR_PALETTE, numColors)

    # @staticmethod
    # def setDecimalPoints(val: int) -> None:
    #     DECIMAL_POINTS = val

    # @staticmethod
    # def setNumBins(val: int) -> None:
    #     NUM_BINS = val

    # @staticmethod
    # def setRetentionRate(val: float) -> None:
    #     RETENTION_RATE = val

    # @staticmethod
    # def setConversionLoss(val: float) -> None:
    #     CONVERSION_LOSS = val
