import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def discretize_variable(series, bins=3):
    return pd.cut(series, bins=bins, labels=['Low', 'Medium', 'High'])

def build_bayesian_network(df):
    bn_df = pd.DataFrame()
    bn_df['Stock_Price_Level'] = discretize_variable(df['Stock_Price'])
    bn_df['Volume_Level'] = discretize_variable(df['Volume'])
    bn_df['Returns_Level'] = discretize_variable(df['Returns'])
    bn_df['Volatility_Level'] = discretize_variable(df['Volatility'])
    bn_df = bn_df.dropna()

    model = BayesianNetwork([
        ('Volume_Level', 'Stock_Price_Level'),
        ('Returns_Level', 'Stock_Price_Level'),
        ('Volatility_Level', 'Returns_Level'),
        ('Volume_Level', 'Volatility_Level')
    ])
    model.fit(bn_df, estimator=MaximumLikelihoodEstimator)
    inference = VariableElimination(model)
    return model, inference
