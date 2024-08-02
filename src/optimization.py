import gc
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
from strategy.moving_average import MovingAverage
from strategy.z_score import ZScore
from strategy.rsi import RSI
from strategy.roc import ROC
from strategy.percentile import Percentile
from strategy.min_max import MinMax
from strategy.robust import Robust
from joblib import Parallel, delayed
from memory_profiler import profile

class Optimization():

    def __init__(self, strategy_name, source_df, window_size_list, threshold_list, target, price='Price', long_short='long', condition='higher'):
        self.strategy_name = strategy_name
        self.source_df = source_df
        self.window_size_list = window_size_list
        self.threshold_list = threshold_list
        self.target = target
        self.price = price
        self.long_short = long_short
        self.condition = condition

        self.results_data_df = pd.DataFrame()

        if self.strategy_name not in globals():
            return

    def _run_strategy(self, window_size, threshold):
        strategy = globals()[self.strategy_name]
        result = strategy(self.source_df, window_size, threshold, target=self.target, price=self.price, long_short=self.long_short, condition=self.condition)
        return result

    def run(self):
        results_data = []
        results = Parallel(n_jobs=4)(delayed(self._run_strategy)(window_size, threshold) for window_size in self.window_size_list for threshold in self.threshold_list)

        for result in results:
            result_data = result.dump_data()
            result_data['Strategy Object'] = result

            results_data.append(result_data)

        self.results_data_df = pd.DataFrame(results_data)
        self.results_data_df = self.results_data_df.sort_values(by='Sharpe', ascending=False)

    def plot_heat_map(self):
        result_data_pivot = self.results_data_df.pivot(index='Window', columns='Threshold', values='Sharpe')
        sns.heatmap(result_data_pivot, cmap="Greens", annot=True).set_title(self.strategy_name + " " + self.long_short)
        print(self.results_data_df.head())
        plt.show()

    def get_best(self):

        return self.results_data_df.iloc[0]['Strategy Object']


