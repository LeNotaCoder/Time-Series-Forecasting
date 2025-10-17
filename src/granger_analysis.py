from statsmodels.tsa.stattools import grangercausalitytests, adfuller

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] < 0.05  # True if stationary

def granger_causality(df, target='Stock_Price', maxlag=5):
    variables = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility']
    granger_results = {}

    for var in variables:
        if var in df.columns:
            test_data = df[[target, var]].dropna()
            try:
                gc_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                p_values = [gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
                min_p_value = min(p_values)
                best_lag = p_values.index(min_p_value) + 1
                granger_results[var] = {
                    'min_p_value': min_p_value,
                    'best_lag': best_lag,
                    'is_causal': min_p_value < 0.05
                }
            except:
                granger_results[var] = {'min_p_value': 1.0, 'best_lag': 0, 'is_causal': False}

    causal_vars = [var for var, result in granger_results.items() if result['is_causal']]
    return granger_results, causal_vars
