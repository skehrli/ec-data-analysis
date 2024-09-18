Wrapper for datasets about Energy Communities providing the ability to immediately
extract key stats such as the breakdown of covered energy (self-coverage, trading over market,
discharged from community battery and bought from grid). The market is implicit and computed
optimally at each timestep.
The functionality is provided by the *ECDataset* class. For its construction, provide it with two 2D Pandas DataFrames *production* and *consumption*, where *production*[i][j] is the production of community member j at timestep i and consumption[i][j] is the consumption of community member j at timestep i.

```python

```

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
