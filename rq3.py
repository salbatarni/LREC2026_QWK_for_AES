from metrics import qwk_score, gwet_ac2_linear_score, gwet_ac2_quadratic_score, krippendorff_alpha_score, rmse_score, acc_score, cohen_d_score, pearson_score
import sklearn.metrics as sklearn_metric
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import numpy as np
import pickle
from prmse import prmse_true as prmse
import warnings
from itertools import combinations_with_replacement, combinations
import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
from irrCAC.raw import CAC
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def generate_systems_vectorized(df, percentages, unique_labels=None):
    """
    Generate predicted scores for multiple correct_percentage values at once.
    
    Returns a dict mapping percentage string to a DataFrame copy with 'predicted_score'.
    """
    df = df.copy()
    total_n = len(df)
    labels = df['true_score'].to_numpy()
    
    if unique_labels is None:
        unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Map labels to integers for fast indexing
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = np.array(unique_labels)
    label_indices = np.vectorize(label_to_idx.get)(labels)

    results = {}

    for pct in percentages:
        n_correct = int(pct * total_n)

        # Pick correct indices
        correct_indices = np.random.choice(total_n, size=n_correct, replace=False)
        all_indices = np.arange(total_n)
        incorrect_indices = np.setdiff1d(all_indices, correct_indices)

        predicted_idx = label_indices.copy()

        if len(incorrect_indices) > 0:
            # Random integers from 0..n_labels-2
            rand_idx = np.random.randint(0, n_labels - 1, size=len(incorrect_indices))
            # Shift to avoid true label
            true_idx = label_indices[incorrect_indices]
            rand_idx += (rand_idx >= true_idx)
            predicted_idx[incorrect_indices] = rand_idx

        df_copy = df.copy()
        df_copy['predicted_score'] = idx_to_label[predicted_idx]
        results[f"{int(pct*100)}_percent"] = df_copy

    return results


def seat_combinations(num_seats, step):
    # total number of step units (e.g., for step=10, total_units = 10)
    total_units = 100 // step  
    
    # generate all compositions of total_units into num_seats positive parts
    # using the stars-and-bars approach
    results = []
    for dividers in combinations(range(1, total_units), num_seats - 1):
        parts = []
        last = 0
        for d in dividers:
            parts.append(d - last)
            last = d
        parts.append(total_units - last)
        # multiply by the step to get actual seat values (e.g., 10, 20, 30, …)
        results.append([p * step for p in parts])
    
    return results

def calculate_kendall_tau(metric, baseline_systems, conditioned_systems, min, max):
    def get_scores(systems):
        return {
            system: metric(df['true_score'], df['predicted_score'], min=min, max=max)
            for system, df in systems.items()
        }

    baseline_scores = get_scores(baseline_systems)
    conditioned_scores = get_scores(conditioned_systems)

    # Ensure consistent system order
    systems = list(baseline_scores.keys())
    baseline_ranks = [sorted(baseline_scores.values(), reverse=True).index(baseline_scores[s]) for s in systems]
    conditioned_ranks = [sorted(conditioned_scores.values(), reverse=True).index(conditioned_scores[s]) for s in systems]
    
    tau, _ = kendalltau(baseline_ranks, conditioned_ranks)

    return tau

def generate_df(arr, total):
    if len(arr) != total:
        print('array and total are not the same')
        return ValueError
    return pd.DataFrame({'essay_id': range(total), 'true_score': arr})


def sample_with_min_unique(scores, size, p, min_unique=2, max_tries=100):
    for _ in range(max_tries):
        sample = np.random.choice(scores, size=size, p=p)
        if len(np.unique(sample)) >= min_unique:
            return sample
    raise ValueError(f"Could not sample with at least {min_unique} unique scores after {max_tries} attempts")

def get_taus(metric, n_samples, runs, uniform_e_df, p_current, scores=np.array([1, 2, 3, 4, 5])):

    # Sample from each
    current_entropy_data = sample_with_min_unique(scores, size=n_samples, p=p_current)
    current_e_df = generate_df(current_entropy_data, n_samples)

    systems_range = np.arange(0.0, 1, 0.02)

    taus = []
    for i in range(runs):
        syn_uniform = generate_systems_vectorized(uniform_e_df, systems_range, unique_labels=scores)
        syn_current = generate_systems_vectorized(current_e_df, systems_range, unique_labels=scores)

        tau = calculate_kendall_tau(metric, syn_uniform, syn_current, min=scores.min(), max=scores.max())

        if not np.isnan(tau):
            taus.append(tau)
    
    return taus


# Score range
scores = np.array([1, 2, 3, 4, 5])
step = 5
runs = 50
n_samples = [1000]


# generate uniform distribution for scores
p_uniform = [1/len(scores)] * len(scores)

metrics = {
    'QWK': qwk_score,
    "Pearson's r": pearson_score,
    'Krippendorff': krippendorff_alpha_score,
    'RMSE': rmse_score,
    'Accuracy': acc_score,
    'AC2 quad': gwet_ac2_quadratic_score,
    'AC2 linear': gwet_ac2_linear_score,
    
    # 'cohen_kappa': sklearn_metric.cohen_kappa_score,
    # "Cohen's d": cohen_d_score,
}

probs = seat_combinations(len(scores), step=step)

results_dict = {}
for metric_name, metric in metrics.items():
    results_dict[metric_name] = {}
    print('-------------------------',metric_name,'-------------------------')
    for sample in n_samples:
        results_dict[metric_name][sample] = {}
        print('Samples:', sample)
        all_taus = []
        # Sample from each
        uniform_entropy_data = sample_with_min_unique(scores, size=sample, p=p_uniform)
        uniform_e_df = generate_df(uniform_entropy_data, sample)
        for i, p_current in enumerate(probs):
            start_time = time.time()
            p_current = np.array(p_current)/100
            ent = stats.entropy(p_current, base=2)
            taus = get_taus(metric, sample, runs, uniform_e_df=uniform_e_df, p_current=p_current, scores=scores)
            all_taus.append(taus)
            end_time = time.time()
            if i % 100 == 0:
                print(f'({i+1}/{len(probs)}) | p_current: {p_current} | Entropy: {ent:.2f} | Time: {end_time - start_time:.2f}s')
        
            # Store lists of taus for this sample size
            results_dict[metric_name][sample][tuple(p_current)] = {
                "entropy": ent,
                "mean_tau": np.mean(taus),
                "std_tau": np.std(taus),
                "taus": taus,
            }

        file_path = f"results_pickels/RQ3/{metric_name}_{len(scores)}scores_{runs}run_{sample}samples_{step}step_entropy_taus.pkl"
        try:
            with open(file_path, "wb") as f:
                pickle.dump(results_dict, f)
        except FileExistsError:
            pass