# 20260610_Project_StockMarket_App
Multimodal B3 Asset Predictor & Alpha Screener. A high-performance, deep learning framework designed to process tabular time-series data and financial text indicators from the Brazilian Stock Exchange (B3). The architecture pairs a sequence-gated Long Short-Term Memory (LSTM) network with a textual Transformer branch (FinBERT base) to generate multi-step autoregressive price forecasts and identify tactical structural long/short market setups.

## 🚀 Key Framework Milestones

### 1. Data Processing Pipeline (ID 001 - ID 004)
* **Robust Local Ingestion**: Imports B3 historical datasets tracking metadata profiles across 489 active listings.
* **Corporate Action Lookback Cleaning**: Resolves raw `#N/A` text sequences and standardizes Brazilian decimal formatting variants (`","` → `"."`). Implements a reverse-chronological `bfill` lookup engine to patch recent data gaps while preserving original timelines.
* **Schema Optimization**: Drops row identifiers and filters layout columns to establish string integer offset coordinates terminating at column `'0'` (representing today's closing price).

### 2. Multimodal Feature Tokenization & Alignment (ID 005 - ID 009)
* **Context String Aggregation**: Merges isolated corporate variables into single, structured row tokens. This process optimizes training passes and enforces fixed tensor footprints (`[489, 306]`) using the specialized **ProsusAI/FinBERT** token layout maps.
* **Leakage-Free Validation Splitting**: Executes a row-synchronized `train_test_split` utilizing a strict random seed mapping vector. This locks identical shuffling permutations across both the textual embedding tensors and the absolute price grids, guaranteeing that asset rows remain perfectly synchronized across splits (**391 Train Assets / 98 Eval Assets**).

### 3. Structural Normalization & Macro Fusion (ID 010 - ID 016)
* **Isolated Row-Wise Scaling**: Replaces global scaling models with asset-isolated local Z-score parameter maps (`dim=1`), preserving individual asset wave forms while stripping absolute pricing magnitude imbalances.
* **Gradient Protection Layers**: Integrates a symmetrical Winsorization/clipping filter, bounding extreme outliers precisely at ±3.5 standard deviations.
* **Hardware-Accelerated Macro Ingestion**: Leverages `yfinance` and the Central Bank of Brazil (`bcb/sgs`) API layers to collect 14 synchronized benchmark channels (including IBOV, EWZ, USD_BRL, CDI, and IPCA). Expands and broadcasts these 2D market matrices into clean 3D feature spaces to construct the final master feature blocks (`[Batch, 60, 15]`), separating the 61st day as a leak-free target variable.

### 4. Neural Network Core Architecture (ID 017 - ID 021)
* **Text branch Transformer**: Embeds long text variables, applies dual custom Multi-Head Attention blocks, and uses global average token pooling to drop the textual dimension down to a compact 32-feature vector space.
* **Time-Series Recurrent Branch**: Implements a step-wise Feature Attention Gating network to dynamically mask and weight absolute prices against the 14 macro signals, feeding a deep 2-layer LSTM projected down to 48-dimensional sequence slots.
* **Downstream Fusion Regressor Head**: Stacks the parallel branches along the feature channel axis (`32 + 48 = 80`), passes the combined vectors through an advanced Multi-Layer Perceptron (MLP) containing Layer Normalization arrays, and employs specialized uniform Xavier weight initializations to optimize model gradients.

### 5. Production Optimization & Backtesting Telemetry (ID 022 - ID 026)
* **Batch Execution System**: Trains using an `AdamW` optimizer configured with a `CosineAnnealingLR` decay schedule over 100 epochs, shrinking mean squared error (MSE) loss paths down to `0.0078`.
* **Out-of-Sample Performance Results**: Passes validation tests with high efficiency, recording an out-of-sample **R² score of 0.6909** and achieving a **directional trend precision rate of 89.80%**.
* **Weight Serialization**: Detaches the optimized internal weight states from active Jupyter memory lines and compiles them into standalone, deployment-ready `.pt` files.

### 6. Quantitative Tactical Screening Radars (ID 027 - ID 034)
* **Gating Attention Importance Mapping**: Directly queries model weights to verify macro indicator prioritization, using dynamic color-coding routines to evaluate the primary price anchor against macro drivers.
* **Vectorized Autoregressive Forecasting**: Loops over a 10-step forward horizon using a row-aligned pointer array to correctly reverse individual stock standardizations. This resolves index-shuffle anomalies and prints real BRL currency projections.
* **Technical Analytics Overlays**: Approximates 14-period Relative Strength Indexes (RSI) using `torch.clamp` vectors and calculates 20/50 Simple Moving Average (SMA) spreads over rolling 5-day windows.
* **Geometric Polar Momentum Engine**: Executes an on-device parallel OLS linear regression across all 489 listings, converting time-series trends into smooth geometric angular headings between -90° and +90° via `torch.atan2`.
* **Strategic Screener Integration**: Groups indicator metrics into precise alpha filters to generate the final execution tables:
  * `buy_in_table`: Targets oversold setups displaying positive angular turnarounds and high expected 5-day returns.
  * `sell_out_table`: Deploys a multi-stage relaxation radar that adapts constraints dynamically to flag overbought trend tops.

---

## 📊 Core Data Architecture Ledger

| Tensor Target | Dimension Blueprint | Data Type | Structural Function Mapping |
| :--- | :--- | :--- | :--- |
| `details_full_tensor` | `[489, 306]` | `torch.long` | Tokenized corporate text identifier sequences |
| `master_train_3d_tensor` | `[391, 60, 15]` | `torch.float32` | 391 Training assets fused with 14 macro benchmarks |
| `master_eval_3d_tensor` | `[98, 60, 15]` | `torch.float32` | 98 Validation assets fused with 14 macro benchmarks |
| `price_std_target_train_tensor`| `[391, 1]` | `torch.float32` | Normalized, leak-free Day 61 training target close |
| `price_std_target_eval_tensor` | `[98, 1]` | `torch.float32` | Normalized, leak-free Day 61 validation target close |
| `sma_spread_master_tensor` | `[489, 5, 1]` | `torch.float32` | Trailing 5-day 20/50 SMA relative trend spreads |
| `outputs` / `predictions` | `[Batch, 1]` | `torch.float32` | Forecasted Day 61 price velocity z-scores |

---

## 🛠️ Production Deployment Framework

To restore the trained network parameters and execute inference across the complete B3 portfolio inside an independent environment, instantiate the master class template and reload the serialized state dictionary:

```python
import torch

# 1. Establish structural hyperparameter sizes matching model configuration
model = StockProfitabilityPredictor(
    vocab_size=30522, details_length=306, embedding_dim=64, output_dim=32, num_heads=8,
    lstm_input_size=15, lstm_hidden_size=128, num_layers=2, dropout=0.2
)

# 2. Safely restore serialized weights directly into model tracking layers
export_path = "./production_models/stock_predictor_90pc_accuracy.pt"
model.load_state_dict(torch.load(export_path, map_location=torch.device('cpu')))
model.eval()

print("🎯 Multimodal B3 Predictor initialized successfully for real-time asset screening.")
```

