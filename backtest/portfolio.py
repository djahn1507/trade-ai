from math import sqrt

def _coerce_to_float(value):
    """Convert pandas/numpy scalars without triggering FutureWarnings."""

    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass

    if hasattr(value, "iloc"):
        try:
            first_value = value.iloc[0]
        except (IndexError, TypeError, AttributeError):
            first_value = value
        else:
            return _coerce_to_float(first_value)

    return float(value)


def _flatten_to_float_list(values):
    """Converts nested iterables into a flat list of floats."""
    if isinstance(values, (list, tuple)):
        flattened = []
        for item in values:
            flattened.extend(_flatten_to_float_list(item))
        return flattened
    if hasattr(values, "tolist"):
        return _flatten_to_float_list(values.tolist())
    return [_coerce_to_float(values)]


def kapital_backtest(
    df,
    predictions,
    threshold=0.5,
    initial_cash=10_000,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    probability_buffer: float = 0.0,
    slippage_pct: float = 0.0005,
    trading_fee_pct: float = 0.0005,
    cooldown_bars: int = 1,
):
    """Simuliert eine Portfolio-Strategie auf Basis von Modellvorhersagen."""
    assert 'Close' in df.columns, "'Close'-Spalte fehlt im DataFrame"
    assert len(predictions) + 1 <= len(df), (
        "DataFrame ist zu kurz oder predictions zu lang!")

    cash = float(initial_cash)
    position = 0.0
    equity_curve = []
    trades = []
    total_fees = 0.0

    total_trades = 0
    winning_trades = 0
    losing_trades = 0

    if hasattr(predictions, 'values'):
        predictions = predictions.values

    predictions = _flatten_to_float_list(predictions)

    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    cooldown_counter = 0

    def _determine_position_size(confidence: float) -> float:
        if confidence > 0.7:
            return 1.0
        if confidence > 0.6:
            return 0.8
        if confidence > 0.5:
            return 0.6
        if confidence > 0.4:
            return 0.4
        return 0.3

    for i in range(1, len(predictions) + 1):
        price_index = min(i, len(df) - 1)
        price_today = float(df['Close'].iloc[price_index])

        if cooldown_counter > 0:
            cooldown_counter -= 1

        pred_value = float(predictions[i - 1])
        signal_bool = pred_value >= (threshold + probability_buffer)

        if position > 0:
            effective_exit_price = price_today * (1 - slippage_pct)
            if price_today <= stop_price:
                proceeds = position * effective_exit_price
                fee = proceeds * trading_fee_pct
                cash += proceeds - fee
                total_fees += fee
                trade_result = (effective_exit_price / entry_price - 1) * 100
                trades.append({
                    "Art": "Verkauf (Stop-Loss)",
                    "Einstiegspreis": entry_price,
                    "Ausstiegspreis": effective_exit_price,
                    "Rendite_pct": trade_result,
                    "Gebühr": fee,
                })
                losing_trades += 1
                total_trades += 1
                position = 0.0
                entry_price = 0.0
                cooldown_counter = max(cooldown_counter, cooldown_bars)
            elif price_today >= target_price:
                proceeds = position * effective_exit_price
                fee = proceeds * trading_fee_pct
                cash += proceeds - fee
                total_fees += fee
                trade_result = (effective_exit_price / entry_price - 1) * 100
                trades.append({
                    "Art": "Verkauf (Take-Profit)",
                    "Einstiegspreis": entry_price,
                    "Ausstiegspreis": effective_exit_price,
                    "Rendite_pct": trade_result,
                    "Gebühr": fee,
                })
                winning_trades += 1
                total_trades += 1
                position = 0.0
                entry_price = 0.0
                cooldown_counter = max(cooldown_counter, cooldown_bars)

        if signal_bool and position == 0 and cash > 0 and cooldown_counter == 0:
            position_fraction = _determine_position_size(pred_value)
            invest_amount = cash * position_fraction

            if invest_amount > 0:
                effective_entry_price = price_today * (1 + slippage_pct)
                if effective_entry_price <= 0:
                    continue
                shares = invest_amount / effective_entry_price
                gross_cost = shares * effective_entry_price
                fee = gross_cost * trading_fee_pct
                total_cost = gross_cost + fee
                if total_cost > cash:
                    shares = cash / (effective_entry_price * (1 + trading_fee_pct))
                    gross_cost = shares * effective_entry_price
                    fee = gross_cost * trading_fee_pct
                    total_cost = gross_cost + fee

                if shares <= 0:
                    continue

                cash -= total_cost
                total_fees += fee
                position = shares
                entry_price = effective_entry_price

                if 'atr' in df.columns:
                    atr_index = min(i, len(df) - 1)
                    atr_value = float(df['atr'].iloc[atr_index])
                    dynamic_sl = atr_value * 2
                    stop_price = max(0.0, entry_price - dynamic_sl)
                    dynamic_tp = atr_value * 4
                    target_price = entry_price + dynamic_tp
                else:
                    stop_price = max(0.0, entry_price * (1 - stop_loss_pct))
                    target_price = entry_price * (1 + take_profit_pct)

        elif not signal_bool and position > 0 and entry_price > 0:
            effective_exit_price = price_today * (1 - slippage_pct)
            proceeds = position * effective_exit_price
            fee = proceeds * trading_fee_pct
            cash += proceeds - fee
            total_fees += fee
            trade_result = (effective_exit_price / entry_price - 1) * 100
            trades.append({
                "Art": "Verkauf (Signal)",
                "Einstiegspreis": entry_price,
                "Ausstiegspreis": effective_exit_price,
                "Rendite_pct": trade_result,
                "Gebühr": fee,
            })

            if price_today > entry_price:
                winning_trades += 1
            else:
                losing_trades += 1

            total_trades += 1
            position = 0.0
            entry_price = 0.0
            cooldown_counter = max(cooldown_counter, cooldown_bars)

        portfolio_value = cash + position * price_today
        equity_curve.append(portfolio_value)

    if position > 0:
        price_today = float(df['Close'].iloc[-1])
        effective_exit_price = price_today * (1 - slippage_pct)
        proceeds = position * effective_exit_price
        fee = proceeds * trading_fee_pct
        cash += proceeds - fee
        total_fees += fee

        if entry_price > 0:
            trade_result = (effective_exit_price / entry_price - 1) * 100
            trades.append({
                "Art": "Verkauf (Backtest-Ende)",
                "Einstiegspreis": entry_price,
                "Ausstiegspreis": effective_exit_price,
                "Rendite_pct": trade_result,
                "Gebühr": fee,
            })

            if price_today > entry_price:
                winning_trades += 1
            else:
                losing_trades += 1

            total_trades += 1

        position = 0.0
        entry_price = 0.0

    peak = []
    current_max = float('-inf')
    for value in equity_curve:
        current_max = max(current_max, value)
        peak.append(current_max)

    if peak and equity_curve:
        drawdown = [
            ((p - eq) / p * 100) if p else 0.0
            for p, eq in zip(peak, equity_curve)
        ]
        max_drawdown = max(drawdown) if drawdown else 0.0
    else:
        drawdown = []
        max_drawdown = 0.0

    if len(equity_curve) > 1:
        daily_returns = []
        for prev, curr in zip(equity_curve[:-1], equity_curve[1:]):
            if prev != 0:
                daily_returns.append((curr - prev) / prev)

        if daily_returns:
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
            std_return = variance ** 0.5
            sharpe_ratio = (mean_return / std_return * sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
    final_value = equity_curve[-1] if equity_curve else initial_cash
    gesamtrendite = (final_value - initial_cash) / initial_cash * 100

    return {
        "Startkapital": initial_cash,
        "Endkapital": round(final_value, 2),
        "Rendite (%)": round(gesamtrendite, 2),
        "Max Equity": round(max(equity_curve), 2) if equity_curve else initial_cash,
        "Min Equity": round(min(equity_curve), 2) if equity_curve else initial_cash,
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Anzahl Trades": total_trades,
        "Gewonnene Trades": winning_trades,
        "Verlorene Trades": losing_trades,
        "Win-Rate": round(win_rate * 100, 2),
        "Handelsgebühren": round(total_fees, 2),
        "Trade-Details": trades,
        "Equity-Verlauf": equity_curve,
    }
