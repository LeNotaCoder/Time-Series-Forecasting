from pytorch_lightning import seed_everything
from src.data_loader import load_and_prepare_data
from src.granger_analysis import granger_causality
from src.bayesian_network import build_bayesian_network
from src.tft_model import prepare_tft_dataset, train_tft
from src.evaluation import evaluate_model
from src.visualization import plot_results


seed_everything(42, workers=True)
df = load_and_prepare_data("./data/ALPN.csv")

# Granger causality
granger_results, causal_vars = granger_causality(df)
print(f"Causal variables: {causal_vars}")

# Bayesian network
model, inference = build_bayesian_network(df)
print(f"Bayesian Network nodes: {list(model.nodes())}, edges: {list(model.edges())}")

# Prepare TFT dataset
training, validation, train_dataloader, val_dataloader = prepare_tft_dataset(df, causal_vars)

# Train TFT
tft, train_losses, val_losses = train_tft(training, train_dataloader, val_dataloader)

# Evaluate
median_predictions, all_actuals, mae, rmse, mape = evaluate_model(tft, val_dataloader)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

# Visualization
plot_results(train_losses, val_losses, median_predictions, all_actuals, granger_results, df, model)
