"""PyTorch LSTM forecaster with windowing, Huber loss, and static SKU embeddings.

Implements P2-08 & TRD §5.1:
- Global model, PyTorch, CPU execution
- Windowing: 56-day input window -> 28-day target output
- Step features: scaled log1p(units), day-of-week sin/cos, holiday flag, price ratio
- Static SKU entity embeddings (dim 8)
- Huber loss on log1p scale with expm1 inversion clipped at >= 0.0
- Early stopping mechanism (patience 5)
- Deterministic execution with fixed seeds
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from retailpulse.features.forecast_features import clip_training_targets
from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.base import Forecaster, resolve_target_dates

logger = get_logger(__name__)


@dataclass(frozen=True)
class LSTMWindowData:
    """Tensors representing windowed sequence features and targets."""

    x_seq: np.ndarray[Any, Any]  # Shape: (N, window_size, 5) float32
    sku_indices: np.ndarray[Any, Any]  # Shape: (N,) int64
    y_target: np.ndarray[Any, Any]  # Shape: (N, horizon) float32 in log1p scale
    dates: list[datetime.date]  # Target start dates


class DemandLSTMNet(nn.Module):
    """Sequence-to-vector LSTM network with static entity embeddings."""

    def __init__(
        self,
        num_skus: int,
        *,
        embedding_dim: int = 8,
        step_feature_dim: int = 5,
        hidden_size: int = 64,
        num_layers: int = 1,
        horizon: int = 28,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # +1 for unknown / out-of-vocabulary SKUs
        self.sku_embedding = nn.Embedding(
            num_embeddings=num_skus + 1,
            embedding_dim=embedding_dim,
        )
        lstm_input_dim = step_feature_dim + embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x_seq: torch.Tensor, sku_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding, LSTM, and projection head."""
        batch_size, seq_len, _ = x_seq.shape
        emb = self.sku_embedding(sku_idx)  # (B, embedding_dim)
        emb_expanded = emb.unsqueeze(1).expand(batch_size, seq_len, -1)  # (B, T, embedding_dim)

        lstm_in = torch.cat([x_seq, emb_expanded], dim=2)  # (B, T, 5 + embedding_dim)
        _lstm_out, (hn, _cn) = self.lstm(lstm_in)
        last_hidden = hn[-1]  # (B, hidden_size)
        out = self.fc(last_hidden)
        return cast(torch.Tensor, out)


def _compute_step_features(
    dates: list[datetime.date],
    y_values: np.ndarray[Any, Any],
    prices: np.ndarray[Any, Any],
    holidays: np.ndarray[Any, Any],
    *,
    log1p_mean: float,
    log1p_std: float,
) -> np.ndarray[Any, Any]:
    """Compute (T, 5) step features: [scaled_log1p, dow_sin, dow_cos, holiday, price_ratio]."""
    n = len(dates)
    features = np.zeros((n, 5), dtype=np.float32)

    # 1. Scaled log1p units
    log1p_y = np.log1p(np.maximum(0.0, y_values))
    denom = log1p_std if log1p_std > 1e-5 else 1.0
    features[:, 0] = (log1p_y - log1p_mean) / denom

    # 2. Day of week sin / cos
    dows = np.array([d.isocalendar()[2] - 1 for d in dates], dtype=float)
    features[:, 1] = np.sin(2.0 * np.pi * dows / 7.0)
    features[:, 2] = np.cos(2.0 * np.pi * dows / 7.0)

    # 3. Holiday flag
    features[:, 3] = holidays.astype(np.float32)

    # 4. Price ratio relative to median
    median_p = float(np.median(prices)) if len(prices) > 0 else 1.0
    if median_p > 0:
        features[:, 4] = np.clip(prices / median_p, 0.1, 10.0)
    else:
        features[:, 4] = 1.0

    return features


def build_lstm_windows(
    df: pd.DataFrame,
    *,
    sku_to_idx: dict[Any, int],
    window_size: int = 56,
    horizon: int = 28,
    stride_days: int = 1,
    log1p_mean: float = 0.0,
    log1p_std: float = 1.0,
) -> LSTMWindowData:
    """Extract rolling windows of (56-day inputs, 28-day targets) across all series."""
    x_seq_list: list[np.ndarray[Any, Any]] = []
    sku_list: list[int] = []
    y_target_list: list[np.ndarray[Any, Any]] = []
    target_dates_list: list[datetime.date] = []

    for sid, group in df.groupby("series_id"):
        sorted_g = group.sort_values("date").reset_index(drop=True)
        n_obs = len(sorted_g)
        if n_obs < window_size + horizon:
            continue

        dates = list(sorted_g["date"])
        y_vals = sorted_g["y"].to_numpy(dtype=float)
        prices = (
            sorted_g["price"].to_numpy(dtype=float)
            if "price" in sorted_g.columns
            else np.ones(n_obs)
        )
        hols = (
            sorted_g["is_holiday"].to_numpy(dtype=bool)
            if "is_holiday" in sorted_g.columns
            else np.zeros(n_obs, dtype=bool)
        )

        step_feats = _compute_step_features(
            dates=dates,
            y_values=y_vals,
            prices=prices,
            holidays=hols,
            log1p_mean=log1p_mean,
            log1p_std=log1p_std,
        )

        log1p_targets = np.log1p(np.maximum(0.0, y_vals)).astype(np.float32)
        sku_code = sku_to_idx.get(sid, len(sku_to_idx))

        max_start = n_obs - window_size - horizon + 1
        for start_idx in range(0, max_start, stride_days):
            in_end = start_idx + window_size
            tgt_end = in_end + horizon

            x_seq_list.append(step_feats[start_idx:in_end])
            sku_list.append(sku_code)
            y_target_list.append(log1p_targets[in_end:tgt_end])
            target_dates_list.append(dates[in_end])

    if not x_seq_list:
        return LSTMWindowData(
            x_seq=np.empty((0, window_size, 5), dtype=np.float32),
            sku_indices=np.empty(0, dtype=np.int64),
            y_target=np.empty((0, horizon), dtype=np.float32),
            dates=[],
        )

    return LSTMWindowData(
        x_seq=np.stack(x_seq_list, axis=0).astype(np.float32),
        sku_indices=np.array(sku_list, dtype=np.int64),
        y_target=np.stack(y_target_list, axis=0).astype(np.float32),
        dates=target_dates_list,
    )


class LSTMForecaster(Forecaster):
    """Global multi-series LSTM demand forecaster implemented in PyTorch (CPU)."""

    name: str = "lstm"

    def __init__(
        self,
        *,
        window_size: int = 56,
        horizon: int = 28,
        hidden_size: int = 64,
        embedding_dim: int = 8,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 30,
        early_stopping_patience: int = 5,
        stride_days: int = 1,
        clip_training_percentile: float | None = 0.995,
        random_seed: int = 42,
    ) -> None:
        self.window_size = window_size
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.stride_days = stride_days
        self.clip_training_percentile = clip_training_percentile
        self.random_seed = random_seed

        self.device = torch.device("cpu")
        self._model: DemandLSTMNet | None = None
        self._history: pd.DataFrame | None = None
        self._calendar_df: pd.DataFrame | None = None
        self._sku_to_idx: dict[Any, int] = {}
        self._log1p_mean: float = 0.0
        self._log1p_std: float = 1.0
        self._training_history: list[dict[str, float]] = []

    @property
    def training_history(self) -> list[dict[str, float]]:
        """Training loss history per epoch."""
        return list(self._training_history)

    def _set_seeds(self) -> None:
        """Enforce strict seed reproducibility for deterministic CPU execution."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)  # noqa: NPY002
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _train_epochs(
        self,
        train_loader: DataLoader[Any],
        x_val: torch.Tensor,
        sku_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> None:
        """Execute iterative training loop with early stopping."""
        if self._model is None:
            return

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_counter = 0
        self._training_history.clear()

        for epoch in range(1, self.max_epochs + 1):
            self._model.train()
            train_losses: list[float] = []

            for bx, b_sku, by in train_loader:
                batch_x = bx.to(self.device)
                batch_sku = b_sku.to(self.device)
                batch_y = by.to(self.device)

                optimizer.zero_grad()
                pred = self._model(batch_x, batch_sku)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(x_val.to(self.device), sku_val.to(self.device))
                val_loss = float(criterion(val_pred, y_val.to(self.device)).item())

            self._training_history.append(
                {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss}
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        "lstm early stopping triggered",
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                    )
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

    def fit(
        self,
        history: pd.DataFrame,
        calendar_df: pd.DataFrame | None = None,
    ) -> LSTMForecaster:
        """Train global LSTM model using 56-to-28 day rolling windows.

        Args:
            history: Historical sales DataFrame containing:
                series_id, date, y, price, is_holiday, abc_class.
            calendar_df: Optional dim_date DataFrame.

        Returns:
            Fitted LSTMForecaster instance.
        """
        if history.empty:
            raise ValueError("Cannot fit LSTMForecaster on empty history DataFrame")

        self._set_seeds()
        self._history = history.copy()
        self._calendar_df = calendar_df

        train_hist = history
        if self.clip_training_percentile is not None and "y" in history.columns:
            train_hist = clip_training_targets(
                history,
                target_col="y",
                group_col="series_id",
                quantile=self.clip_training_percentile,
            )

        # Build SKU dictionary
        unique_skus = sorted(train_hist["series_id"].unique())
        self._sku_to_idx = {sku: idx for idx, sku in enumerate(unique_skus)}

        # Global scaling parameters
        log1p_vals = np.log1p(np.maximum(0.0, train_hist["y"].to_numpy(dtype=float)))
        self._log1p_mean = float(np.mean(log1p_vals)) if len(log1p_vals) > 0 else 0.0
        self._log1p_std = float(np.std(log1p_vals)) if len(log1p_vals) > 0 else 1.0

        # Extract rolling windows
        windows = build_lstm_windows(
            train_hist,
            sku_to_idx=self._sku_to_idx,
            window_size=self.window_size,
            horizon=self.horizon,
            stride_days=self.stride_days,
            log1p_mean=self._log1p_mean,
            log1p_std=self._log1p_std,
        )

        n_samples = len(windows.x_seq)
        if n_samples == 0:
            total_days = (history["date"].max() - history["date"].min()).days
            raise ValueError(
                f"Insufficient history span ({total_days} days) for window_size={self.window_size} "
                f"and horizon={self.horizon}"
            )

        # Train/Validation split:
        # Per TRD §5.1: validation on the last 28 days of each training fold
        val_cutoff_date = train_hist["date"].max() - datetime.timedelta(days=self.horizon)
        val_mask = np.array([d >= val_cutoff_date for d in windows.dates])

        if np.sum(val_mask) > 0 and np.sum(~val_mask) > 0:
            train_indices = np.where(~val_mask)[0]
            val_indices = np.where(val_mask)[0]
        else:
            # Fallback for very small datasets: 80% train, 20% val
            split_point = max(1, int(n_samples * 0.8))
            train_indices = np.arange(split_point)
            val_indices = np.arange(split_point, n_samples)
            if len(val_indices) == 0:
                val_indices = train_indices  # Self-eval if single window

        x_train = torch.from_numpy(windows.x_seq[train_indices])
        sku_train = torch.from_numpy(windows.sku_indices[train_indices])
        y_train = torch.from_numpy(windows.y_target[train_indices])

        x_val = torch.from_numpy(windows.x_seq[val_indices])
        sku_val = torch.from_numpy(windows.sku_indices[val_indices])
        y_val = torch.from_numpy(windows.y_target[val_indices])

        train_dataset = TensorDataset(x_train, sku_train, y_train)
        effective_batch_size = min(self.batch_size, len(train_dataset))
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=False,  # deterministic order
        )

        # Initialize network
        self._model = DemandLSTMNet(
            num_skus=len(self._sku_to_idx),
            embedding_dim=self.embedding_dim,
            step_feature_dim=5,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            horizon=self.horizon,
        ).to(self.device)

        self._train_epochs(train_loader, x_val, sku_val, y_val)

        logger.info(
            "fitted lstm forecaster",
            total_windows=n_samples,
            epochs_run=len(self._training_history),
            final_train_loss=self._training_history[-1]["train_loss"],
        )
        return self

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for future horizon dates strictly after origin.

        Args:
            origin: Last observed date of the training/inference cutoff.
            horizon: Number of trading days ahead to forecast.
            calendar_df: Optional dim_date DataFrame.

        Returns:
            DataFrame with columns: [series_id, product_id, date, horizon_day, yhat, abc_class]
            where yhat >= 0.0.
        """
        if self._model is None or self._history is None:
            raise ValueError("LSTMForecaster must be fitted before predict() can be called")

        effective_cal = calendar_df if calendar_df is not None else self._calendar_df
        target_dates = resolve_target_dates(origin, horizon, effective_cal)
        if not target_dates:
            return pd.DataFrame(
                columns=["series_id", "product_id", "date", "horizon_day", "yhat", "abc_class"]
            )

        self._model.eval()
        hist = self._history[self._history["date"] <= origin].copy()
        if hist.empty:
            return pd.DataFrame(
                columns=["series_id", "product_id", "date", "horizon_day", "yhat", "abc_class"]
            )

        results: list[pd.DataFrame] = []

        for sid, group in hist.groupby("series_id"):
            sorted_g = group.sort_values("date").reset_index(drop=True)
            if sorted_g.empty:
                continue

            prod_id = sorted_g["product_id"].iloc[-1]
            abc_class = sorted_g["abc_class"].iloc[-1] if "abc_class" in sorted_g.columns else "C"

            # Extract last window_size observations, left-padding if shorter
            tail_g = sorted_g.tail(self.window_size)
            n_tail = len(tail_g)

            dates = list(tail_g["date"])
            y_vals = tail_g["y"].to_numpy(dtype=float)
            prices = (
                tail_g["price"].to_numpy(dtype=float)
                if "price" in tail_g.columns
                else np.ones(n_tail)
            )
            hols = (
                tail_g["is_holiday"].to_numpy(dtype=bool)
                if "is_holiday" in tail_g.columns
                else np.zeros(n_tail, dtype=bool)
            )

            step_feats = _compute_step_features(
                dates=dates,
                y_values=y_vals,
                prices=prices,
                holidays=hols,
                log1p_mean=self._log1p_mean,
                log1p_std=self._log1p_std,
            )

            if n_tail < self.window_size:
                pad_len = self.window_size - n_tail
                pad = np.repeat(step_feats[:1], pad_len, axis=0)
                step_feats = np.vstack([pad, step_feats])

            sku_code = self._sku_to_idx.get(sid, len(self._sku_to_idx))

            x_tensor = torch.from_numpy(step_feats).unsqueeze(0).to(self.device)  # (1, 56, 5)
            sku_tensor = torch.tensor([sku_code], dtype=torch.int64).to(self.device)

            with torch.no_grad():
                out_log1p = self._model(x_tensor, sku_tensor).cpu().numpy()[0]  # (horizon_model,)

            # Invert log1p and enforce non-negative constraint
            yhat_full = np.maximum(0.0, np.expm1(out_log1p))
            yhat = yhat_full[:horizon]

            df_pred = pd.DataFrame(
                {
                    "series_id": sid,
                    "product_id": prod_id,
                    "date": target_dates[: len(yhat)],
                    "horizon_day": list(range(1, len(yhat) + 1)),
                    "yhat": yhat,
                    "abc_class": abc_class,
                }
            )
            results.append(df_pred)

        if not results:
            return pd.DataFrame(
                columns=["series_id", "product_id", "date", "horizon_day", "yhat", "abc_class"]
            )

        combined = pd.concat(results, ignore_index=True)
        return combined.sort_values(["series_id", "date"]).reset_index(drop=True)
