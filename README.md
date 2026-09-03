# Transitive DIF Clustering (TDC) Simulation Reproduction

This repository contains the code and classification-probability inputs needed to reproduce the Transitive DIF Clustering (TDC) simulation analyses for the Ten-group and Three-group conditions.

The repository generates MIRT and Stan constraint files from previously produced InterDIFNet classification probabilities. It does not include the empirical-data analysis, trained InterDIFNet models, or the upstream model-training workflow.

## Repository contents

```text
TDC.py
run_tdc.py
Optimal_Thresholds_Ten.csv
Optimal_Thresholds_Three.csv
InterDIFNet_Ten_Group_Simulation_Study_Results/
InterDIFNet_Three_Group_Simulation_Study_Results/
```

- `TDC.py` contains the TDC analysis functions.
- `run_tdc.py` runs the included simulation conditions.
- The threshold CSVs contain the decision thresholds used by TDC.
- The two InterDIFNet directories contain the classification-probability inputs.

## Included simulation cases

The intended simulation design contains 600 cases for each group condition:

- Two DIF percentages: 20% and 40%
- Three sample sizes
- Up to 100 replications per condition

The included data contain:

| Design | Sample sizes | Available cases |
|---|---|---:|
| Ten groups | 1,000; 2,000; 4,000 | 558 |
| Three groups | 250; 500; 1,000 | 586 |

Fifty-six intended replications are unavailable because their upstream estimation procedures did not converge:

- 42 Ten-group cases
- 14 Three-group cases

`run_tdc.py` discovers the available replication files separately for each simulation condition. It therefore processes every included case without assuming that all replication numbers are available.

## Python requirements

Python 3.9 or later is recommended.

The code imports the following packages:

```text
numpy
pandas
tensorflow
scikit-learn
matplotlib
seaborn
scikit-multilearn
networkx
scipy
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install numpy pandas tensorflow scikit-learn matplotlib seaborn scikit-multilearn networkx scipy
```

Installing the dependencies before running the analysis avoids the interactive dependency-installation prompt in `TDC.py`.

## Running the simulations

From the repository directory, run:

```bash
python run_tdc.py
```

The runner resolves its paths relative to its own location, so it may also be invoked from another working directory:

```bash
python /path/to/TDC_repository/run_tdc.py
```

The simulations can take substantial time to complete.

## Generated output

The runner creates these directories:

```text
TDC_Ten_Group_Results/
TDC_Three_Group_Results/
```

For each available replication, the analysis generates:

- MIRT constraint syntax in a text file
- Stan discrimination-parameter constraints in CSV format
- Stan difficulty-parameter constraints in CSV format
- Stan constraint metadata in JSON format

Plots are disabled by default to reduce runtime and storage requirements. Generated result directories are excluded by `.gitignore`.

## Analysis settings

The runner uses:

- DIF types: `DIF_a` and `DIF_b`
- Saved MIRT constraints: enabled
- Saved Stan constraints: enabled
- TDC plots: disabled
- K-means plots: disabled
- Hierarchical-clustering plots: disabled
- Interactive plot display: disabled

The Ten- and Three-group analyses use their corresponding top-level threshold files.

## Reproducing individual conditions

The primary callable function is:

```python
from TDC import TDC_simulation_study
```

Example:

```python
from pathlib import Path
from TDC import TDC_simulation_study

repository = Path(__file__).resolve().parent

TDC_simulation_study(
    groups="Three",
    percentages=[20],
    sizes=[250],
    replications=[1],
    dif_types=["DIF_a", "DIF_b"],
    data_folder=str(
        repository / "InterDIFNet_Three_Group_Simulation_Study_Results"
    ),
    results_folder=str(repository / "TDC_Three_Group_Results"),
    thresholds_file=str(repository / "Optimal_Thresholds_Three.csv"),
    save_mirt_constraints=True,
    save_stan_constraints=True,
    generate_tdc_plots=False,
    generate_kmeans_plots=False,
    generate_hierarchical_plots=False,
    show_plots=False,
    verbose=False,
)
```

Only request a replication when its corresponding classification CSV is present.

## Scope

This repository reproduces the downstream TDC clustering stage from supplied InterDIFNet classification probabilities. It does not reproduce:

- InterDIFNet training
- Generation of the original simulation data
- Empirical-data analyses
- Standalone K-means or hierarchical analyses
- Subsequent psychometric parameter estimation

## Citation and license

Add the study citation and an appropriate software/data license before publicly distributing this repository.
