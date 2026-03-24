import numpy as np
import pandas as pd
import pickle
from metrics import *

def num_categories_exp(metric, systems, syn_systems, group=1, ellipse=False):

  def rescale_scores(scores, old_min, old_max, new_min, new_max):
    return ((scores - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

  groups = {
    1: {
      'tasks': [2, 3, 4, 5, 6],
      'num_cat': [2, 3, 4, 5],
      'range': [0, 1, 2, 3, 4]
    },
    2: {
      'tasks': [1, 7, 8],
      'num_cat': [2, 3, 4, 5, 6, 7, 8, 9],
      'range': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
  }

  if ellipse:
      groups = {
        1: {
          'tasks': list(range(0,44)),
          'num_cat': [2, 3, 4, 5, 6, 7, 8, 9],
          'range': [2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
      }

  systems_performance = {}
  num_categories = groups[group]['num_cat']

  # Define the full range of scores to standardize on (e.g., [2, 10])
  range_min = groups[group]['range'][0]
  range_max = groups[group]['range'][-1]

  for num_cat in num_categories:
    systems_performance[num_cat] = {}
    for system in systems:
      df = syn_systems[system]

      df = df[df['prompt_id'].isin(groups[group]['tasks'])] 
      
      # First, normalize all scores to same range either [2, 10] or [0, 5]
      if not ellipse:
        for prompt_id in df['prompt_id'].unique():
          if prompt_id in [3, 4]:
            continue
          mask = df['prompt_id'] == prompt_id
          # Rescale true_score
          true_scores = df.loc[mask, 'true_score']
          true_min = true_scores.min()
          true_max = true_scores.max()
          rescaled_true = rescale_scores(true_scores, true_min, true_max, range_min, range_max).round()
          df.loc[mask, 'true_score'] = rescaled_true
          # Rescale predicted_score
          pred_scores = df.loc[mask, 'predicted_score']
          rescaled_pred = rescale_scores(pred_scores, true_min, true_max, range_min, range_max).round()
          df.loc[mask, 'predicted_score'] = rescaled_pred

      # Normalize true_score to [range_min, num_cat - 1]
      min_score = groups[group]['range'][0]
      max_score = groups[group]['range'][num_cat-1]
      true_min = int(df['true_score'].min())
      true_max = int(df['true_score'].max())
      df['true_score'] = rescale_scores(df['true_score'], true_min, true_max, min_score, max_score)
      df['true_score'] = df['true_score'].round().astype(int)
      df['predicted_score'] = rescale_scores(df['predicted_score'], true_min, true_max, min_score, max_score)
      df['predicted_score'] = df['predicted_score'].round().astype(int)

      # Calculate QWK for each prompt and get average
      metric_value = df.groupby('prompt_id').apply(lambda x: metric(x['true_score'], x['predicted_score'], min=min_score, max=max_score), include_groups=False).mean()
      systems_performance[num_cat][system] = metric_value


  # Store Kendall's tau results
  kendall_scores = {}

  # Get the baseline ranking at 100%
  baseline_scores = systems_performance[num_categories[-1]]
  baseline_ranking = sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True)
  baseline_order = [system for system, _ in baseline_ranking]

  # Loop through other num_cat values and compute Kendall’s tau
  for num_cat in num_categories[:-1]:
      current_scores = systems_performance[num_cat]
      current_ranking = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
      current_order = [system for system, _ in current_ranking]

      # Ensure both lists are in the same order (based on baseline systems)
      baseline_rank = [baseline_order.index(system) for system in baseline_order]
      current_rank = [current_order.index(system) for system in baseline_order]

      tau, _ = kendalltau(baseline_rank, current_rank)
      kendall_scores[num_cat] = tau

  # Add perfect score for num_cat=5 itself
  kendall_scores[num_categories[-1]] = 1.0

  return kendall_scores, systems_performance


def main():
    with open("systems_pickels/ELLIPSE_50_syn_systems_allPrompts.pkl", "rb") as f:
        syn_systems = pickle.load(f)


    metrics = {
        'QWK': qwk_score,
        # 'cohen_kappa': sklearn_metric.cohen_kappa_score,
        "Cohen's d": cohen_d_score,
        "Pearson's r": pearson_score,
        'Krippendorff': krippendorff_alpha_score,
        'RMSE': rmse_score,
        'Accuracy': acc_score,
        'AC2 quad': gwet_ac2_quadratic_score,
        'AC2 linear': gwet_ac2_linear_score,
    }

    xs = []
    ys = []
    metric_performance = {}
    for name, metric_fn in metrics.items():
        avg_kendall_scores, systems_performance = num_categories_exp(metric_fn, systems=list(syn_systems.keys()), group=1, ellipse=True)
        metric_performance[name] = systems_performance

        # Add 100% with tau = 1.0
        x = sorted(avg_kendall_scores.keys())
        x = sorted(x)
        y = [avg_kendall_scores.get(size, 1.0) for size in x]  # Default 1.0 at 100%
        xs.append(x)
        ys.append(y)


    # Save variables to a file
    # with open('results_pickels/RQ2_1/group_2.pkl', 'wb') as f:
    with open('results_pickels/ELLIPSE/RQ2_1.pkl', 'wb') as f:
        pickle.dump({
            'metric_performance': metric_performance,
            'xs': xs,
            'ys': ys
        }, f)