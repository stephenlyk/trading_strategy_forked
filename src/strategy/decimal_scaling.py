# decimal_scaling.py
import numpy as np
import modin.pandas as pd
from strategy.strategy import Strategy
import logging

logger = logging.getLogger(__name__)


class DecimalScaling(Strategy):
    def __init__(self, metric, source_df, window_size, threshold, target='Target', price='Price', long_short="long",
                 condition="higher"):
        super().__init__(metric, source_df, window_size, threshold, target=target, price=price, long_short=long_short,
                         condition=condition)
        self.result_df = self._decimal_scaling_strategy(source_df.copy(), window_size, threshold, target, long_short,
                                                        condition)
        self.annual_return = Strategy.annual_return(self.result_df)
        self.mdds = Strategy.return_mdds(self.result_df['Cumulative_Profit'])
        self.mdd = self.mdds[self.mdds.last_valid_index()]
        self.calmar = self.annual_return / abs(self.mdd)
        self.sharpe = Strategy.get_sharpe(self.result_df)

    def _decimal_scaling_strategy(self, df, window_size, threshold, target, long_short, condition):
        # Calculate max absolute value in rolling window
        rolling_max = df[target].abs().rolling(window=int(window_size)).max()

        # Calculate scaling factors (number of digits)
        scaling_factors = np.floor(np.log10(rolling_max) + 1)

        # Apply decimal scaling
        df['Decimal_Scaled'] = df[target] / (10 ** scaling_factors)

        # Normalize to [0,1] range in rolling window
        df['rolling_min'] = df['Decimal_Scaled'].rolling(window=int(window_size)).min()
        df['rolling_max'] = df['Decimal_Scaled'].rolling(window=int(window_size)).max()
        df['Decimal_Scaled'] = (df['Decimal_Scaled'] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'])

        # Handle NaN values using ffill() instead of fillna(method='ffill')
        df['Decimal_Scaled'] = df['Decimal_Scaled'].ffill().fillna(0.5)

        self._add_position(df, "Decimal_Scaled", "bounded", threshold, long_short, condition)
        return df