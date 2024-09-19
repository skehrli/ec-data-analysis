#!/usr/bin/env python3

from typing import TypeAlias

NetworkAlloc: TypeAlias = dict[str, dict[str, float]]

# flow network
SOURCE = "s"
BATTERY = "b"
TARGET = "t"
UNBOUNDED = float("inf")

# decimal points to round to
DECIMAL_POINTS = 2

# number of bins for bar charts
N_BINS = 100

# numerical error margin
EPS = 1e-9

# battery retention rate per timestep
RETENTION_RATE = 0.995
# conversion loss for every transaction with battery
CONVERSION_LOSS = 0.05
