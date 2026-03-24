# Unified RQ2 Script (ASAP + ELLIPSE)

from metrics import qwk_score, gwet_ac2_linear_score, gwet_ac2_quadratic_score, krippendorff_alpha_score, rmse_score, acc_score, cohen_d_score, pearson_score
import pandas as pd
from scipy.stats import kendalltau
import numpy as np
import pickle
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


# ======================
# DATASET CONFIGS
# ======================
DATASET_CONFIGS = {
    "ASAP": {
        "data_path": "/data/home/shared/Arabic_Dataset/predictions/ASAP_AES_aug_predictions.csv",
        "groups": {
            1: {
                'tasks': [2, 3, 4, 5, 6],
                'range': [0, 1, 2, 3, 4]
            },
            2: {
                'tasks': [1, 7, 8],
                'range': [2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
        },
        "prompt_ranges": {
            1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3),
            5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60),
        },
        "needs_prompt_rescale": True,
        "systems_path": "systems_pickels/50_syn_systems.pkl",
        "output_dir": "results_pickels/RQ2"
    },
    "ELLIPSE": {
        "data_path": "/data/ellipse_cleaned_17prompts.csv",
        "groups": {
            1: {
                'tasks': list(range(0, 44)),
                'range': [2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
        },
        "prompt_ranges": (2, 10),
        "needs_prompt_rescale": False,
        "systems_path": "systems_pickels/ELLIPSE_50_syn_systems_17prompts.pkl",
        "output_dir": "results_pickels/ELLIPSE/RQ2"
    }
}


# ======================
# NORMALIZATION
# ======================
def normalize_scores(df, config, group, num_cat):
    def rescale(scores, old_min, old_max, new_min, new_max):
        return ((scores - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    group_info = config["groups"][group]
    min_score = group_info['range'][0]
    max_score = group_info['range'][num_cat - 1]

    if config["needs_prompt_rescale"]:
        for prompt_id in df['prompt_id'].unique():
            if prompt_id in [3, 4]:
                continue
            mask = df['prompt_id'] == prompt_id
            old_min, old_max = config["prompt_ranges"][prompt_id]

            df.loc[mask, 'true_score'] = rescale(df.loc[mask, 'true_score'], old_min, old_max, min_score, max_score)
            df.loc[mask, 'predicted_score'] = rescale(df.loc[mask, 'predicted_score'], old_min, old_max, min_score, max_score)
    else:
        old_min, old_max = config["prompt_ranges"]
        df['true_score'] = rescale(df['true_score'], old_min, old_max, min_score, max_score)
        df['predicted_score'] = rescale(df['predicted_score'], old_min, old_max, min_score, max_score)

    df['true_score'] = df['true_score'].round().astype(int)
    df['predicted_score'] = df['predicted_score'].round().astype(int)

    return df


# ======================
# CORE FUNCTION
# ======================
def num_cat_vs_dataset_size(syn_systems, config, runs, sample_sizes, metric, group, num_cat, x_axis):

    all_kendall_scores = {size: [] for size in sample_sizes}
    sample_sizes = sample_sizes + ['full_set']

    for run in range(runs):
        if run % 10 == 0:
            print(f"Run {run+1}/{runs}")

        systems_performance = {}

        for sample_size in sample_sizes:
            systems_performance[sample_size] = {}

            df_ids = pd.read_csv(config["data_path"])
            df_ids = df_ids[df_ids['prompt_id'].isin(config["groups"][group]['tasks'])]

            if sample_size != 'full_set':
                if x_axis == "percentage":
                    df_ids = df_ids.groupby('prompt_id').apply(
                        lambda x: x.sample(int(len(x)*sample_size/100), replace=False)
                    ).reset_index(drop=True)
                else:
                    df_ids = df_ids.groupby('prompt_id').apply(
                        lambda x: x.sample(min(sample_size, len(x)), replace=False)
                    ).reset_index(drop=True)

            sampled_ids = df_ids['essay_id'].tolist()

            for system, df in syn_systems.items():
                df_sampled = df[df['essay_id'].isin(sampled_ids)].copy()

                df_sampled = normalize_scores(df_sampled, config, group, num_cat)

                min_score = config["groups"][group]['range'][0]
                max_score = config["groups"][group]['range'][num_cat - 1]

                metric_performance = df_sampled.groupby('prompt_id').apply(
                    lambda x: metric(x['true_score'], x['predicted_score'], min=min_score, max=max_score),
                    include_groups=False
                ).mean()

                systems_performance[sample_size][system] = metric_performance

        baseline_scores = systems_performance['full_set']
        baseline_order = [s for s, _ in sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True)]

        for sample_size in sample_sizes:
            if sample_size == 'full_set':
                continue

            current_scores = systems_performance[sample_size]
            current_order = [s for s, _ in sorted(current_scores.items(), key=lambda x: x[1], reverse=True)]

            baseline_rank = [baseline_order.index(system) for system in baseline_order]
            current_rank = [current_order.index(system) for system in baseline_order]

            tau, _ = kendalltau(baseline_rank, current_rank)
            all_kendall_scores[sample_size].append(tau)

    avg_kendall = {size: np.mean(all_kendall_scores[size]) for size in all_kendall_scores}
    std_kendall = {size: np.std(all_kendall_scores[size]) for size in all_kendall_scores}

    return avg_kendall, std_kendall


# ======================
# RUN EXPERIMENT
# ======================
def run_experiment(dataset_name="ASAP"):
    config = DATASET_CONFIGS[dataset_name]

    with open(config["systems_path"], "rb") as f:
        syn_systems = pickle.load(f)

    metrics = {
        'QWK': qwk_score,
        "Pearson": pearson_score,
        'Krippendorff': krippendorff_alpha_score,
        'RMSE': rmse_score,
        'Accuracy': acc_score,
        'AC2 quad': gwet_ac2_quadratic_score,
        'AC2 linear': gwet_ac2_linear_score,
    }

    runs = 50
    sample_sizes = list(range(10, 110, 10)) if dataset_name == "ELLIPSE" else list(range(10, 210, 10))
    x_axis = 'sample'

    for metric_name, metric_fn in metrics.items():
        print(f"\nMetric: {metric_name}")

        xs, ys, yerrs = [], [], []

        for num_cat in range(2, 10):
            print(f"Categories: {num_cat}")
            start = time.time()

            avg, std = num_cat_vs_dataset_size(
                syn_systems, config, runs, sample_sizes, metric_fn,
                group=1, num_cat=num_cat, x_axis=x_axis
            )

            print(f"Time: {time.time() - start:.2f}s")

            x = sorted(avg.keys())
            y = [avg[k] for k in x]
            yerr = [std[k] for k in x]

            xs.append(x)
            ys.append(y)
            yerrs.append(yerr)

        out_path = f"{config['output_dir']}/{metric_name}_{runs}runs_kendall.pkl"
        with open(out_path, "wb") as f:
            pickle.dump((xs, ys, yerrs), f)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_experiment("ASAP")
    # run_experiment("ELLIPSE")
