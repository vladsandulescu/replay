import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot(success_rates):
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.plot(success_rates.index, success_rates["success_dqn"].values, linestyle='--', color='red', label='DQN')
    plt.plot(success_rates.index, success_rates["success_her"].values, linestyle='-', color='blue', label='DQN + HER')
    plt.xticks(np.arange(1, max(success_rates.index) + 1, 1.0))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend(loc=1, bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.xlabel('bits')
    plt.ylabel('success rate')
    sns.despine()
    plt.show()

results_dqn = pd.DataFrame.from_csv('experiments/results_dqn.csv')
results_her = pd.DataFrame.from_csv('experiments/results_her.csv')
success_rates = pd.merge(results_dqn, results_her, on=["n", "trial"], suffixes=("_dqn", "_her")). \
                    groupby("n")["success_dqn", "success_her"].mean()
plot(success_rates)