import torch
import numpy as np

def evaluate_model(tft, val_dataloader):
    tft.eval()
    predictions_list, actuals_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            x, y = batch
            output = tft(x)
            pred = output["prediction"] if isinstance(output, dict) else output
            predictions_list.append(pred)
            actuals_list.append(y[0])

    all_predictions = torch.cat(predictions_list, dim=0)
    all_actuals = torch.cat(actuals_list, dim=0)

    median_idx = all_predictions.shape[-1] // 2 if all_predictions.dim() == 3 else 0
    median_predictions = all_predictions[..., median_idx] if all_predictions.dim() == 3 else all_predictions

    mae = torch.mean(torch.abs(median_predictions - all_actuals)).item()
    mse = torch.mean((median_predictions - all_actuals)**2).item()
    rmse = np.sqrt(mse)
    mape = torch.mean(torch.abs((median_predictions - all_actuals) / all_actuals)).item() * 100

    return median_predictions, all_actuals, mae, rmse, mape
