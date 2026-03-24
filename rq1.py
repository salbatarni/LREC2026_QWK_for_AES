from metrics import (
    qwk_score, gwet_ac2_linear_score, gwet_ac2_quadratic_score,
    krippendorff_alpha_score, rmse_score, acc_score,
    cohen_d_score, pearson_score
)

import pandas as pd
from scipy.stats import kendalltau
import numpy as np
import pickle
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


# =========================
# Dataset configurations
# =========================
DATASETS = {
    "ASAP": {
        "data_path": "data/ASAP_100samples_perPrompt.csv",
        "score_range": {
            1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
            5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60),
        },
        "filter_fn": lambda df: df[df['prompt_id'] != 8],
        "systems_path": "systems_pickels/ASAP_50_syn_systems_100samplesPerprompts.pkl",
        "output_path": "results_pickels/ASAP/RQ1/",
    },
    "ELLIPSE": {
        "data_path": "data/ellipse_cleaned_17prompts.csv",
        "score_range": (2, 10),
        "filter_fn": None,
        "systems_path": "systems_pickels/ELLIPSE_50_syn_systems_17prompts.pkl",
        "output_path": "results_pickels/ELLIPSE/RQ1/",
    }
}


# =========================
# Helper
# =========================
def get_min_max(prompt_id, score_range):
    if isinstance(score_range, dict):
        return score_range[prompt_id]
    return score_range


# =========================
# Core function
# =========================
def get_tau_scores(
    syn_systems,
    runs,
    sample_sizes,
    metric,
    data_path,
    score_range,
    x_axis='sample',
    filter_fn=None
):
    all_kendall_scores = {size: [] for size in sample_sizes}
    sample_sizes = sample_sizes + ['full_set']

    for run in range(runs):
        if run % 10 == 0 and run != 0:
            print(f'Completed {run} runs...')

        failed_run = False
        systems_performance = {}

        for sample_size in sample_sizes:
            systems_performance[sample_size] = {}

            df_ids = pd.read_csv(data_path)

            if filter_fn is not None:
                df_ids = filter_fn(df_ids)

            # Sampling
            if sample_size != 'full_set':
                if x_axis == "percentage":
                    df_ids = df_ids.groupby('prompt_id').apply(
                        lambda x: x.sample(int(len(x) * sample_size / 100), replace=False)
                    ).reset_index(drop=True)
                else:
                    df_ids = df_ids.groupby('prompt_id').apply(
                        lambda x: x.sample(min(sample_size, len(x)), replace=False)
                    ).reset_index(drop=True)

            sampled_ids = df_ids['essay_id'].tolist()

            try:
                for system, df in syn_systems.items():
                    df_filtered = df[df['essay_id'].isin(sampled_ids)]

                    metric_performance = df_filtered.groupby('prompt_id').apply(
                        lambda x: metric(
                            x['true_score'],
                            x['predicted_score'],
                            min=get_min_max(x.name, score_range)[0],
                            max=get_min_max(x.name, score_range)[1]
                        ),
                        include_groups=False
                    ).mean()

                    systems_performance[sample_size][system] = metric_performance

            except Exception as e:
                print(f"Skipped run {run}, sample size {sample_size}")
                print("Error:", e)
                failed_run = True
                break

        if failed_run:
            continue

        # Baseline ranking (full set)
        baseline_scores = systems_performance['full_set']
        baseline_order = [
            system for system, _ in sorted(
                baseline_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        # Compare rankings
        for sample_size in sample_sizes:
            if sample_size == 'full_set':
                continue

            current_scores = systems_performance[sample_size]
            current_order = [
                system for system, _ in sorted(
                    current_scores.items(), key=lambda x: x[1], reverse=True
                )
            ]

            baseline_rank = [baseline_order.index(s) for s in baseline_order]
            current_rank = [current_order.index(s) for s in baseline_order]

            tau, _ = kendalltau(baseline_rank, current_rank)
            all_kendall_scores[sample_size].append(tau)

    avg_kendall = {k: np.mean(v) for k, v in all_kendall_scores.items()}
    std_kendall = {k: np.std(v) for k, v in all_kendall_scores.items()}

    return avg_kendall, std_kendall


# =========================
# Main experiment
# =========================
if __name__ == "__main__":

    dataset_name = "ASAP"  # change to "ELLIPSE"
    cfg = DATASETS[dataset_name]

    runs = 50
    sample_sizes = list(range(10, 110, 10))
    x_axis = 'sample'

    metrics = {
        'QWK': qwk_score,
        "Pearson": pearson_score,
        "Krippendorff": krippendorff_alpha_score,
        "RMSE": rmse_score,
        "Accuracy": acc_score,
        "AC2 quad": gwet_ac2_quadratic_score,
        "AC2 linear": gwet_ac2_linear_score,
    }

    # Load systems
    with open(cfg["systems_path"], "rb") as f:
        syn_systems = pickle.load(f)

    xs, ys, yerrs = [], [], []

    for name, metric_fn in metrics.items():
        print(f"Running {dataset_name} - {name}")

        avg_k, std_k = get_tau_scores(
            syn_systems,
            runs,
            sample_sizes,
            metric_fn,
            data_path=cfg["data_path"],
            score_range=cfg["score_range"],
            x_axis=x_axis,
            filter_fn=cfg["filter_fn"]
        )

        x = sorted(avg_k.keys())
        y = [avg_k.get(size, 1.0) for size in x]
        yerr = [std_k[size] for size in x]

        xs.append(x)
        ys.append(y)
        yerrs.append(yerr)

    os.makedirs(cfg["output_path"], exist_ok=True)

    out_file = os.path.join(
        cfg["output_path"],
        f"{runs}runs_kendall_vs_dataset_size.pkl"
    )

    with open(out_file, "wb") as f:
        pickle.dump((xs, ys, yerrs), f)

    print(f"Saved results to {out_file}")