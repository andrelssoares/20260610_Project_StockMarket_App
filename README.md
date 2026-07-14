## Multimodal 3-Day & 5-day Rolling Interval Stock Market Tracking Framework
An advanced, high-performance financial machine learning pipeline designed to predict stock market structural price movements by fusing unstructured text descriptors (FinBERT token sequences), low-frequency micro corporate accounting metrics, and synchronized macroeconomic indicators [ID: 010, ID: 014].
This architecture implements a fast, vectorized, de-fragmented engineering pipeline using contiguous 3D NumPy matrices and native PyTorch attention blocks [ID: 004, ID: 018]. It systematically prevents chronological lookahead information leakage by calculating all scaling and distribution parameters exclusively within past lookback bounds [ID: 011].

## ⚡ Key Architectural Upgrades (Resolving MAE & R² Variance)
To eliminate typical time-series degradation limits (where naive models hit a performance ceiling of R² ≈ 0.5 and an out-of-sample MAE > 0.6), this production-ready implementation deploys three advanced architectural pillars:
1. Learnable Attentive Pooling Layer
   Legacy global average pooling dilutes critical temporal momentum right before the trading execution day. This framework implements a multi-layer learnable attention pooling network:
   \(\text{Scores}_{t}=\mathbf{W}_{2}\cdot \tanh (\mathbf{W}_{1}\cdot \mathbf{X}_{t}+\mathbf{b}_{1})\)
   \(\alpha _{t}=\text{Softmax}(\text{Scores}_{t})\)
   \(\mathbf{X}_{\text{pooled}}=\sum _{t}\alpha _{t}\mathbf{X}_{t}\)
   This enables the network to dynamically weigh high-signal market anomalies or macroeconomic shocks over the 24-step (5-day agent) or 30-step (3-day agent) lookback timeline [ID: 023].

2. Stratified Regularization Blueprint
   Rather than applying a blanket dropout across different data types, this pipeline deploys a specialized multi-branch masking strategy to preserve weak signals while suppressing structural text noise [ID: 017]:
   * NLP Text Embeddings: 0.1 spatial embedding stabilization.
   * NLP Attention Blocks: 0.3 token descriptor regularizer (prevents text memorization).
   * Tabular Branch: 0.0 (Zero Dropout) to keep low-frequency metrics ([LIQUIDITY, EQUITY]) fully visible.
   * LSTM Recurrent Layers: 0.2 variational timeline step masking to filter stock market noise.
   * Feed-Forward Residual Core: 0.4 terminal dropout to force generalized latent abstractions.

 3. Vectorized Memory-Safe Operations
    The pipeline bypasses slow cell-by-cell row iterations by performing right-to-left launch gap stabilization and unscaling loops directly inside contiguous C-speed NumPy blocks, executing at a blazing-fast processing speed of over 24,000 rows per second [ID: 004].

    📂 Multi-Branch Pipeline Graph Layout                        +----------------------------+

                        |  Master Input Asset Pool   |
                        +--------------+-------------+
                                       |
             +-------------------------+-------------------------+

             |                         |                         |
+------------v------------+ +----------v----------+ +------------v------------+

| Unstructured Text Branch| | Static Tabular Branch| | Sequential Market Branch |
|  Corporate Descriptors  | |  [LIQUIDITY, EQUITY] | |   OHLCV Arrays + Macro   |
+------------+------------+ +----------+----------+ +------------+------------+

             |                         |                         |
    FinBERT Tokenizer         Symmetrical Log Scale           nn.LSTM Stack

             |                         |                         |
  Trainable Pos. Embeddings     Linear Projection         Gated Feature Attention

             |                         |                         |
 Stacked Attention Blocks              |                         |

             |                         |                         |
  Attentive Token Pooling              |                         |

             |                         |                         |
     (Batch, 24) Shape         (Batch, 16) Shape        (Batch, Seq, 128) Shape

             |                         |                         |
             +------------+------------+                         |

                          |                                      |
                 Concatenate & Expand                            |

                          |                                      |
               (Batch, Seq, 40) Shape                            |

                          |                                      |
                          +------------------+-------------------+
                                             |
                                    Multimodal Fusion
                                             |
                                  (Batch, Seq, 168) Shape
                                             |
                                  Transformer Attention Block
                                             |
                                   Attentive Pooling Layer
                                             |
                                     (Batch, 168) Shape
                                             |
                                   Deep Feed-Forward Net
                                             |
                                    Linear Projection Head
                                             |
                                  +----------v----------+
                                  | Forward Target Pred |
                                  +---------------------+

## ⚙️ Hyperparameter Configuration Constraints
Programmatically extracted from active dataset partitions to guarantee absolute shape divisibility prior to neural graph allocations [ID: 017]:
* Attention Bottleneck Dimensionality: 168 features total (24 NLP Features + 16 Tabular Features + 128 Time-Series Hidden Channels).
* Parallel Attention Head Count: 8 heads (\(\text{Divisibility sanity check}: 168 \bmod 8 \equiv 0\), yielding a fixed allocation of exactly 21 feature nodes per active attention head).
* Recurrent Deep Grid Layering: 2 stacked LSTM layers mapped to an adaptive feature gating mask [ID: 021].
* Tabular Dimensional Shape: 2 unique channels (LIQUIDITY and EQUITY scaled natively via Symmetrical Log \(\text{sign}(x) \cdot \log(1 + \vert{}x\vert{})\) transformations to track negative corporate net worth without exceptions) [ID: 003, ID: 004].

## 📈 Optimization & Backtest Performance Strategy
The training routine is locked into low-memory mini-batches utilizing AdamW optimization, coupled with a OneCycleLR learning rate scheduler to achieve super-convergence across 50 epochs [ID: 024]. Residual accounting errors and severe price anomalies are handled using a robust Huber Loss cost criterion (δ=1.0) [ID: 024].

## Out-of-Sample Portfolio Strategy
The model's out-of-sample execution weights are validated using a risk-stabilized equal-weighted long/short asset strategy [ID: 027]:
* Long Signals (Pred ≥ 0): Automatically buys the target asset at the trading interval anchor point (Day 91 or Day 121 depending on configuration scale).
* Short Signals (Pred < 0): Execution short positions.
* Penny-Stock Safety Floor: Implements a native financial currency floor (R$ 0.50) to guarantee complete immunity against division-by-zero anomalies in real market environments [ID: 027].


Framework Version Tracker: Unified Multi-Modal Agent Engine (3-Day/5-Day/1-Day Calibrated Variants) [ID: 023]. Locked under random seed 42 to preserve replication integrity across text and timeline blocks [ID: 010].
