import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

def prepare_tft_dataset(df, causal_vars, max_encoder_length=20, max_prediction_length=5):
    training_cutoff = df['time_idx'].max() - max_prediction_length

    time_varying_known = ['time_idx', 'Open', 'High', 'Low', 'Volume', 'MA_5']
    time_varying_unknown = ['Stock_Price', 'Returns', 'Volatility']

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Stock_Price",
        group_ids=["stock_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["stock_id"],
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    batch_size = min(32, max(4, len(training) // 4))
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    return training, validation, train_dataloader, val_dataloader

def train_tft(training, train_dataloader, val_dataloader, num_epochs=30, learning_rate=0.01):
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=5,
        reduce_on_plateau_patience=4,
    )

    tft.train()
    optimizer = torch.optim.Adam(tft.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Training
        epoch_loss, batch_count = 0.0, 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            output = tft(x)
            predictions = output["prediction"] if isinstance(output, dict) else output
            loss = tft.loss(predictions, y[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tft.parameters(), 0.1)
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        train_losses.append(epoch_loss / batch_count)

        # Validation
        tft.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                output = tft(x)
                predictions = output["prediction"] if isinstance(output, dict) else output
                loss = tft.loss(predictions, y[0])
                val_loss += loss.item()
                val_count += 1
        val_losses.append(val_loss / val_count)
        tft.train()

    return tft, train_losses, val_losses
