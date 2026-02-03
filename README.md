Advanced Time Series Forecasting using Transformer with Attention Mechanism

📌 Project Overview

This project implements an advanced deep learning model for multivariate time series forecasting using a custom-built Transformer architecture with self-attention mechanisms. The goal is to capture long-range temporal dependencies and compare its performance against a strong LSTM baseline model.

The project demonstrates:

Custom Transformer Encoder implementation

Multivariate time series forecasting

Hyperparameter configuration

Model comparison (Transformer vs LSTM)

Evaluation using MAE, RMSE, and MAPE

Attention weight extraction for interpretability

🎯 Objectives

Generate a multivariate time series dataset (5 features, 1200 time steps)

Implement a Transformer Encoder from scratch using PyTorch primitives

Train and evaluate a deep learning forecasting model

Compare performance against a standard LSTM baseline

Analyze attention weights for temporal dependency understanding

📊 Dataset Description

The dataset is programmatically generated and contains:

1200 time steps

5 correlated features

Trend component

Multiple seasonal components

Gaussian noise

The forecasting task predicts the next time step value of Feature 1 using previous 30 time steps (sliding window approach).

🏗 Model Architecture
1️⃣ Transformer Model

Components:

Linear Input Projection Layer

Positional Encoding

Multi-Head Self-Attention

Feed-Forward Network

Layer Normalization

Residual Connections

Final Linear Output Layer

Key Hyperparameters:

d_model = 64

n_heads = 4

num_layers = 2

dim_feedforward = 128

Sequence length = 30

Optimizer = Adam

Loss Function = MSELoss

2️⃣ LSTM Baseline

Components:

Single-layer LSTM

Fully Connected Output Layer

This model serves as a benchmark for comparison.

📐 Evaluation Metrics

The following metrics are used:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

These metrics allow a robust comparison between Transformer and LSTM performance.

📁 Project Structure
time_series_transformer/

│

├── main.py              # Complete end-to-end implementation

├── README.md            # Project documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/time-series-transformer.git
cd time-series-transformer


Install dependencies:

pip install torch numpy pandas scikit-learn

▶️ How to Run
python main.py


The script will:

Generate synthetic multivariate data

Create sliding window sequences

Train Transformer model

Train LSTM baseline

Evaluate both models

Print MAE, RMSE, and MAPE scores

📈 Expected Output

Example output:

Training Transformer...
Epoch 1, Loss: ...
...
Transformer Performance
MAE: ...
RMSE: ...
MAPE: ...

Training LSTM...
Epoch 1, Loss: ...
...
LSTM Performance
MAE: ...
RMSE: ...
MAPE: ...


Typically, the Transformer captures long-range dependencies better due to its attention mechanism.

🔎 Attention Mechanism Interpretation

The Transformer model produces attention weights from the Multi-Head Attention layer.

These weights indicate:

Which previous time steps influenced the prediction most

Whether the model focuses on seasonal cycles

Whether long-range dependencies are captured

Example Interpretation:

If higher attention weights appear around lag 50 or lag 100, it indicates the model has learned seasonal periodicity patterns embedded in the dataset.

🧠 Key Concepts Demonstrated

Self-Attention Mechanism

Positional Encoding

Sequence-to-One Forecasting

Deep Learning vs Recurrent Models

Model Interpretability in Time Series

🚀 Future Improvements

Add learning rate scheduler

Perform K-fold cross-validation

Implement early stopping

Add SARIMAX baseline

Visualize attention weights as heatmaps

Deploy model as API

Use real-world financial or sensor datasets

📚 Technologies Used

Python

PyTorch

NumPy

Pandas

Scikit-learn

👩‍💻 Author

Pravina
Deep Learning & Time Series Enthusiast
