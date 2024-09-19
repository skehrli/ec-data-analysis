Wrapper for datasets about Energy Communities providing the ability to immediately
extract key stats such as the breakdown of covered energy (self-coverage, trading over market,
and discharged from community battery) and automatically generate a report. The market is implicit and computed optimally at each timestep using a max-flow algorithm.

The functionality is provided by the *ECDataset* class. For its construction, provide it with two 2D Pandas DataFrames *production* and *consumption*, where *production*[i][j] is the production of community member j at timestep i and consumption[i][j] is the consumption of community member j at timestep i.


Example usage is provided in the ec_data_analysis/main.py file:
```python
def main() -> None:
    data: pd.DataFrame = pd.read_excel("data/EC_EV_dataset.xlsx", sheet_name=None)
    production: pd.DataFrame = getSheet("PV", data)
    consumption: pd.DataFrame = getSheet("Load", data)

    ecData: ECDataset = ECDataset(production, consumption)
    capacities: np.ndarray = np.linspace(1e2, 1e3, 3)
    ecData.createReport(capacities)
```

The example compares the effect on the community of batteries of 3 different capacities (100kw/h, 550kh/h and 1000kw/h). The example report generated by this is in out/report.pdf based on the example dataset data/EC_EV_dataset.xlsx.

The code is annotated with type hints, and the provided Makefile typechecks with

```bash
Make typecheck
```

Run 
```bash
Make all
```
to format the python files, typecheck and run main.py, or
```bash
Make run
```
to simply run main.py.
