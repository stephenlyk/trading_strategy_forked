import numpy as np
import pandas as pd
import plotly.express as px

class Strategy:

    def __init__(self, source_df, window_size, threshold, target='Price', price='Price', long_short="long", condition="higher"):
        self.source_df = source_df
        self.source_df[target] = pd.to_numeric(self.source_df[target], errors='coerce')
        self.source_df['Changes'] = self.source_df['Price'].pct_change(fill_method=None)
        self.source_df['Cumulative_Bnh'] = self.source_df['Changes'].cumsum()
        self.window_size = window_size
        self.threshold = abs(threshold)
        self.target = target

        self.annual_return = 0
        self.mdd = 0
        self.calmar = 0
        self.sharpe = 0

    @staticmethod
    def annual_return(df, profit='Profit'):
        multiplier = Strategy.return_timeframe_multiplier(df)
        return df[profit].mean() * multiplier

    @staticmethod
    def return_moving_average(df, col_name, window_size):
        return df[col_name].rolling(window_size).mean()

    @staticmethod
    def return_moving_average_sd(df, col_name, window_size):
        return df[col_name].rolling(window_size).std()

    @staticmethod
    def get_sharpe(df, profit='Profit'):
        if df[profit].std() != 0 :
            sharpe = (df[profit].mean() / df[profit].std()) * np.sqrt(Strategy.return_timeframe_multiplier(df)).item()
        else:
            sharpe = 0
        return sharpe

    @staticmethod
    def return_mdds(df):
        roll_max = df.cummax()
        epsilon = 1e-9
        roll_max.replace(0, epsilon, inplace=True)
        drawdown = df/roll_max - 1.0
        mdds = drawdown.cummin()
        return mdds

    @staticmethod
    def return_timeframe_multiplier(df):
        diff = df['Date'].diff()
        most_common_interval = diff.mode()[0]
        multiplier = 0
        if most_common_interval >= pd.Timedelta(days=1):
            multiplier = 365
        elif most_common_interval >= pd.Timedelta(hours=1):
            multiplier = 365 * 24
        elif most_common_interval >= pd.Timedelta(minutes=1):
            multiplier = 365 * 24 * 60
        elif most_common_interval >= pd.Timedelta(seconds=1):
            multiplier = 365 * 24 * 60 * 60

        return multiplier

    def _add_position(self, df, signal, signal_type, threshold, long_short, condition):
        match signal_type:
            # signal can be positive and negative
            case "diff":
                match long_short:
                    case "long":
                        if condition == "higher":
                            # buy when price is higher than moving average
                            df['Position'] = (df[signal] > threshold).astype(int)
                        if condition == "lower":
                            # buy  when price is lower than moving average
                            df['Position'] = (df[signal] < (threshold * - 1)).astype(int)

                    case "short":
                        if condition == "higher":
                            # short when price is higher than moving average
                            df['Position'] = (df[signal] > threshold).astype(int) * - 1
                        if condition == "lower":
                            # short when price is lower than moving average
                            df['Position'] = (df[signal] < (threshold * - 1)).astype(int) * -1

                    case "both":
                        if condition == "higher":
                            df['Position'] = (df[signal]  > threshold).astype(int) + \
                                            (df[signal] < (threshold * - 1)).astype(int) * -1
                        if condition == "lower":
                            df['Position'] = (df[signal] < (threshold * -1)).astype(int) + \
                                            (df[signal]  > threshold).astype(int) * -1

            # signal is between 0 and 1
            case "bounded":
                match long_short:
                    case "long":
                        if condition == "higher":
                            df['Position'] = (df[signal] > threshold).astype(int)
                        if condition == "lower":
                            df['Position'] = (df[signal] < threshold).astype(int)

                    case "short":
                        if condition == "higher":
                            df['Position'] = (df[signal] > threshold).astype(int) * -1
                        if condition == "lower":
                            df['Position'] = (df[signal] < threshold).astype(int) * -1
                    case "both":
                        if condition == "higher":
                            df['Position'] = ((df[signal] > threshold).astype(int)) + \
                                            ((df[signal] < (1 - threshold)).astype(int) * -1)
                        if condition == "lower":
                            df['Position'] = ((df[signal] < threshold).astype(int)) + \
                                            ((df[signal] > (1 - threshold)).astype(int) * -1)



        df['Profit'] = df['Position'].shift(1) * df['Changes']
        df['Cumulative_Profit'] = df['Profit'].cumsum()

    def dump_data(self):
        data = {
            'Window': self.window_size,
            'Threshold': self.threshold,
            'Sharpe': self.sharpe,
            'Annual_Return': self.annual_return,
            'MDD': self.mdd,
            'Calmar': self.calmar
        }

        return data

    def plot_graph(self):
        fig = px.line(self.result_df, x='Date', y=['Cumulative_Profit', 'Cumulative_Bnh'], \
                      title=type(self).__name__ + " " + str(self.window_size) + " " + str(self.threshold))
        fig.show()
