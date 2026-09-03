#!/usr/bin/env python3
"""Run the TDC simulation studies using the classification files in this repository."""

from pathlib import Path
import re
from typing import List

from TDC import TDC_simulation_study


REPOSITORY_DIR = Path(__file__).resolve().parent


def discover_replications(
    data_folder: Path,
    groups: str,
    sample_size: int,
    dif_percentage: int,
) -> List[int]:
    """Return sorted replication numbers available for one simulation condition."""
    filename_pattern = re.compile(
        rf"^Classification_Results_{re.escape(groups)}_"
        rf"{sample_size}_{dif_percentage}_Replication(\d+)\.csv$"
    )
    replications = []

    for file_path in data_folder.glob("Classification_Results_*.csv"):
        match = filename_pattern.match(file_path.name)
        if match:
            replications.append(int(match.group(1)))

    return sorted(set(replications))


def run_simulation_design(
    groups: str,
    sample_sizes: List[int],
    data_folder_name: str,
    results_folder_name: str,
    thresholds_filename: str,
) -> int:
    """Run every available sample-size, DIF-percentage, and replication condition."""
    data_folder = REPOSITORY_DIR / data_folder_name
    results_folder = REPOSITORY_DIR / results_folder_name
    thresholds_file = REPOSITORY_DIR / thresholds_filename
    processed_cases = 0

    if not data_folder.is_dir():
        raise FileNotFoundError(f"Simulation data folder not found: {data_folder}")
    if not thresholds_file.is_file():
        raise FileNotFoundError(f"Threshold file not found: {thresholds_file}")

    for sample_size in sample_sizes:
        for dif_percentage in (20, 40):
            replications = discover_replications(
                data_folder=data_folder,
                groups=groups,
                sample_size=sample_size,
                dif_percentage=dif_percentage,
            )
            if not replications:
                print(
                    f"No input files found for {groups} groups, "
                    f"N={sample_size}, DIF={dif_percentage}%; skipping."
                )
                continue

            print(
                f"Running {groups} groups, N={sample_size}, "
                f"DIF={dif_percentage}%: {len(replications)} available replications"
            )
            TDC_simulation_study(
                groups=groups,
                percentages=[dif_percentage],
                sizes=[sample_size],
                replications=replications,
                dif_types=["DIF_a", "DIF_b"],
                data_folder=str(data_folder),
                results_folder=str(results_folder),
                save_mirt_constraints=True,
                save_stan_constraints=True,
                thresholds_file=str(thresholds_file),
                verbose=False,
                generate_tdc_plots=False,
                generate_kmeans_plots=False,
                generate_hierarchical_plots=False,
                show_title=True,
                print_constraints=False,
                color_transitive=True,
                legend_fontsize=20,
                show_plots=False,
            )
            processed_cases += len(replications)

    return processed_cases


def main() -> None:
    ten_group_cases = run_simulation_design(
        groups="Ten",
        sample_sizes=[1000, 2000, 4000],
        data_folder_name="InterDIFNet_Ten_Group_Simulation_Study_Results",
        results_folder_name="TDC_Ten_Group_Results",
        thresholds_filename="Optimal_Thresholds_Ten.csv",
    )
    three_group_cases = run_simulation_design(
        groups="Three",
        sample_sizes=[250, 500, 1000],
        data_folder_name="InterDIFNet_Three_Group_Simulation_Study_Results",
        results_folder_name="TDC_Three_Group_Results",
        thresholds_filename="Optimal_Thresholds_Three.csv",
    )

    print(
        "TDC simulations complete: "
        f"{ten_group_cases} Ten-group cases and "
        f"{three_group_cases} Three-group cases processed."
    )


if __name__ == "__main__":
    main()
