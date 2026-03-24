import numpy as np
import pandas as pd
import pickle

def generate_system(df, correct_percentage, unique_labels=None):
    df = df.copy()
    df['predicted_score'] = np.nan  # initialize the column

    total_n = len(df)
    total_n_correct = int(correct_percentage * total_n)

    # Get the count of samples per prompt
    prompt_counts = df['prompt_id'].value_counts().sort_index()
    total_prompt_counts = prompt_counts.sum()

    # Allocate number of correct predictions per prompt proportionally
    correct_per_prompt = {
        pid: int(count / total_prompt_counts * total_n_correct)
        for pid, count in prompt_counts.items()
    }

    # Adjust for rounding errors (by assigning remaining correct predictions to random prompts)
    remaining = total_n_correct - sum(correct_per_prompt.values())
    if remaining > 0:
        extra_pids = np.random.choice(list(correct_per_prompt.keys()), size=remaining, replace=True)
        for pid in extra_pids:
            correct_per_prompt[pid] += 1

    # Now generate predictions per prompt
    for prompt_id, group in df.groupby('prompt_id'):
        labels = group['true_score'].to_numpy()
        if unique_labels is None:
            unique_labels = np.unique(labels)
        n = len(labels)

        n_correct = correct_per_prompt[prompt_id]
        n_correct = min(n_correct, n)  # safety

        correct_indices = np.random.choice(n, size=n_correct, replace=False)
        incorrect_indices = np.setdiff1d(np.arange(n), correct_indices)

        # Generate predictions
        group_predictions = labels.copy()
        for i in incorrect_indices:
            incorrect_choices = unique_labels[unique_labels != labels[i]]
            group_predictions[i] = np.random.choice(incorrect_choices)

        df.loc[group.index, 'predicted_score'] = group_predictions

    return df


def main():

    df = pd.read_csv('data/ellipse_cleaned_17prompts.csv')
    syn_systems = {f"{int(n*100)}_percent": generate_system(df, n) for n in np.arange(0.0, 1, 0.02)}
    
    try:
        with open("systems_pickels/ELLIPSE_50_syn_systems_17prompts.pkl", "xb") as f:
            pickle.dump(syn_systems, f)
    except FileExistsError:
        pass