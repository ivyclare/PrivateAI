import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy import optimize, stats
from statsmodels.stats.proportion import proportion_confint
import os
import pickle
import argparse


def clopper_pearson(count, trials, conf=0.95):
    q = count / trials
    ci_low = stats.beta.ppf(conf / 2., count, trials - count + 1)
    ci_upp = stats.beta.isf(conf / 2., count + 1, trials - count)

    if np.ndim(ci_low) > 0:
        ci_low[q == 0] = 0
        ci_upp[q == 1] = 1
    else:
        ci_low = ci_low if q != 0 else 0
        ci_upp = ci_upp if q != 1 else 1
    return ci_low, ci_upp

def compute_upper_bounds(fp, fn, num_samples, alpha=0.05):
    # Use clopper_pearson to initially compute upper bounds
    _, initial_alpha_fpr_upper = clopper_pearson(fp, num_samples / 2, alpha)
    _, initial_beta_fnr_upper = clopper_pearson(fn, num_samples / 2, alpha)
    
    # Adjust bounds to avoid -inf or inf from norm.ppf
    fpr_upper, fnr_upper = initial_alpha_fpr_upper, initial_beta_fnr_upper
    if fpr_upper >= 1.0:
        fpr_upper = 1.0 - 1e-10  # Adjust down slightly from 1
    if fnr_upper <= 0:
        fnr_upper = 1e-10  # Adjust up slightly from 0
    
    return fpr_upper, fnr_upper

def compute_rates_from_losses(base_losses, canary_losses, threshold):
    tn = np.sum(base_losses <= threshold)
    fp = np.sum(base_losses > threshold)
    tp = np.sum(canary_losses > threshold)
    fn = np.sum(canary_losses <= threshold)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return fpr, fnr, fp, fn 

def find_epsilon_for_delta(fpr_upper, fnr_upper, delta):
    try:
        mean_diff = norm.ppf(1.0 - fnr_upper) - norm.ppf(fpr_upper)
        # print("Mean Diff: ", mean_diff)
        objective = lambda epsilon: delta - norm.cdf(-epsilon / mean_diff + mean_diff / 2) + np.exp(epsilon) * norm.cdf(-epsilon / mean_diff - mean_diff / 2)
        epsilon_lower_bound = brentq(objective, a=-25, b=25)
        return epsilon_lower_bound
    except ValueError:
        return None  
    
def explore_thresholds(base_losses, canary_losses, delta):
    results = []
    for threshold in np.linspace(np.min(np.concatenate([base_losses, canary_losses])), 
                                 np.max(np.concatenate([base_losses, canary_losses])), 100):
        # print(threshold)
        fpr, fnr, fp, fn = compute_rates_from_losses(base_losses, canary_losses, threshold)
        num_samples = len(base_losses) + len(canary_losses)
        fpr_upper, fnr_upper = compute_upper_bounds(fp, fn, num_samples)
        epsilon_lower_bound = find_epsilon_for_delta(fpr_upper, fnr_upper, delta)
        if epsilon_lower_bound is not None:  # Only add results if epsilon could be calculated
            results.append((threshold, fpr_upper, fnr_upper, epsilon_lower_bound))
    return results

def load_observations(directory_path):
    base_observations = []
    canary_observations = []
    observation_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.startswith('observation') and f.endswith('.pkl')]
    for file_path in observation_files:
        with open(file_path, 'rb') as f:
            base_obs, canary_obs = pickle.load(f)
        base_observations.extend(base_obs)
        canary_observations.extend(canary_obs)
    return base_observations, canary_observations


def main():
    parser = argparse.ArgumentParser(description='Compute lower bounds.')
    parser.add_argument('--gradient_canary', action='store_true', help='Use gradient canary')
    parser.add_argument('--input_canary', action='store_true', help='Use input canary')

    delta = 1e-5
    # with open('observations.pkl', 'rb') as f:
    #     base_observations, canary_observations = pickle.load(f)
    # observation_files = [f for f in os.listdir('./results/') if f.startswith('observation') and f.endswith('.pkl')]
    # # observation_files = [f for f in os.listdir('.') if f.startswith('observation') and f.endswith('.pkl')]
    # print(observation_files)
    # base_observations, canary_observations = load_observations(observation_files)

    
    args = parser.parse_args()
    if args.input_canary: 
        directory_path = './cifar_input_observations/'
    elif args.gradient_canary:
        directory_path = './cifar_grad_observations/'

    base_observations, canary_observations = load_observations(directory_path)
    
    base_losses = np.array(base_observations)  
    canary_losses = np.array(canary_observations) 

    print(f"{base_observations=}")
    print(f"\n\n{canary_observations=}")
    print(f"\nSize of observations: {len(base_losses)}")


    threshold_results = explore_thresholds(base_losses, canary_losses, delta)
    # print(threshold_results)
    df = pd.DataFrame(threshold_results, columns=['Threshold', 'FPR Upper Bound', 'FNR Upper Bound', 'Epsilon Lower Bound'])
    df.to_csv('threshold_results.csv', index=False)
    print(df)

if __name__ == "__main__":
    main()

    