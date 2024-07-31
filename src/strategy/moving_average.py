import numpy as np
import pandas as pd
import plotly.express as px
from strategy.strategy import Strategy

class MovingAverage(Strategy):

    def __init__(self, source_df, window_size, threshold, target='Price', price='Price', long_short="long", condition="higher"):
        super().__init__(source_df, window_size, threshold, target=target, price=price, long_short=long_short, condition=condition)

        self.result_df = self._moving_average_strategy(source_df.copy(), window_size, threshold, target, long_short, condition)
        self.annual_return = Strategy.annual_return(self.result_df)
        self.mdds = Strategy.return_mdds(self.result_df['Cumulative_Profit'])
        self.mdd = self.mdds[self.mdds.last_valid_index()]
        self.calmar = self.annual_return/abs(self.mdd)
        self.sharpe = Strategy.get_sharpe(self.result_df)

    def _moving_average_strategy(self, df, window_size, threshold, target, long_short, condition):
        df['Moving_Average'] = Strategy.return_moving_average(df, target, window_size)
        df['MA_Signal'] = (df[target]/df['Moving_Average']) - 1

        self._add_position(df, "MA_Signal", "diff", threshold, long_short, condition)

        return df

    def plot_graph(self):
        print(self.mdd)
        fig = px.line(self.result_df, x='Date', y='Cumulative_Profit')
        fig.show()
