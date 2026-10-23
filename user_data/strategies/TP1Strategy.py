import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute


class TP1Strategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True
    startup_candle_count = 210

    # الخروج فقط عبر الوقف الهيكلي أو هدف 2R
    minimal_roi = {"0": 100}
    stoploss = -0.99
    use_custom_stoploss = True
    use_exit_signal = False
    process_only_new_candles = True

    RISK_PER_TRADE = 0.005  # 0.5% من الرصيد
    LEVERAGE = 5.0
    ATR_BUFFER = 0.5        # هامش الوقف = 0.5 ATR
    SWING_LEN = 10          # نافذة القاع/القمة
    PULLBACK_LEN = 5        # عدد الشموع للبحث عن التراجع
    PULLBACK_ATR = 0.25     # لمس EMA20 بهامش 0.25 ATR
    MIN_STOP_PCT = 0.003
    MAX_STOP_PCT = 0.05

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # الشمعة المغلقة الحالية معلومة وقت الإشارة، فلا حاجة لـ shift
        swing_low = dataframe["low"].rolling(self.SWING_LEN).min()
        swing_high = dataframe["high"].rolling(self.SWING_LEN).max()
        dataframe["stop_long"] = swing_low - self.ATR_BUFFER * dataframe["atr"]
        dataframe["stop_short"] = swing_high + self.ATR_BUFFER * dataframe["atr"]

        dataframe["pullback_long"] = (
            dataframe["low"].rolling(self.PULLBACK_LEN).min()
            <= dataframe["ema20"] + self.PULLBACK_ATR * dataframe["atr"]
        )
        dataframe["pullback_short"] = (
            dataframe["high"].rolling(self.PULLBACK_LEN).max()
            >= dataframe["ema20"] - self.PULLBACK_ATR * dataframe["atr"]
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dist_long = (dataframe["close"] - dataframe["stop_long"]) / dataframe["close"]
        dist_short = (dataframe["stop_short"] - dataframe["close"]) / dataframe["close"]

        long_cond = (
            (dataframe["close"] > dataframe["ema200"])
            & (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema50_4h"] > dataframe["ema200_4h"])
            & dataframe["pullback_long"]
            & (dataframe["close"] > dataframe["ema20"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["rsi"] > 45)
            & (dataframe["rsi"] < 70)
            & (dist_long > self.MIN_STOP_PCT)
            & (dist_long < self.MAX_STOP_PCT)
            & (dataframe["volume"] > 0)
        )
        short_cond = (
            (dataframe["close"] < dataframe["ema200"])
            & (dataframe["ema50"] < dataframe["ema200"])
            & (dataframe["ema50_4h"] < dataframe["ema200_4h"])
            & dataframe["pullback_short"]
            & (dataframe["close"] < dataframe["ema20"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["rsi"] < 55)
            & (dataframe["rsi"] > 30)
            & (dist_short > self.MIN_STOP_PCT)
            & (dist_short < self.MAX_STOP_PCT)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "long_pullback")
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "short_pullback")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ---------- أدوات مساعدة ----------
    def _signal_row(self, pair: str, when):
        """آخر شمعة مغلقة قبل شمعة الدخول (شمعة الإشارة)."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        t = timeframe_to_prev_date(self.timeframe, when)
        c = df.loc[df["date"] < t]
        return None if c.empty else c.iloc[-1]

    def _stop_price(self, pair: str, is_short: bool, when):
        row = self._signal_row(pair, when)
        if row is None:
            return None
        s = row["stop_short"] if is_short else row["stop_long"]
        return None if np.isnan(s) else float(s)

    # ---------- الرافعة وحجم الصفقة ----------
    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return min(self.LEVERAGE, max_leverage)

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        stop = self._stop_price(pair, side == "short", current_time)
        if stop is None:
            return proposed_stake
        dist = abs(current_rate - stop) / current_rate
        if dist <= 0:
            return proposed_stake
        equity = self.wallets.get_total_stake_amount()
        position_value = equity * self.RISK_PER_TRADE / dist
        stake = position_value / leverage
        stake = min(stake, max_stake)
        if min_stake is not None:
            stake = max(stake, min_stake)
        return stake

    # ---------- الوقف والهدف ----------
    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        stop = self._stop_price(pair, trade.is_short, trade.open_date_utc)
        if stop is None:
            return None
        return stoploss_from_absolute(
            stop, current_rate, is_short=trade.is_short, leverage=trade.leverage
        )

    def custom_roi(self, pair, trade, current_time, trade_duration,
                   entry_tag, side, **kwargs):
        stop = self._stop_price(pair, trade.is_short, trade.open_date_utc)
        if stop is None:
            return None
        risk_pct = abs(trade.open_rate - stop) / trade.open_rate
        return 2.0 * risk_pct * trade.leverage  # هدف 2R
