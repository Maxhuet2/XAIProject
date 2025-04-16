# 🧠 XAI Project – Interpretable LSTM for Financial Time Series Forecasting

This project explores how to make Long Short-Term Memory (LSTM) networks interpretable for the task of forecasting stock prices. It combines built-in interpretability through attention mechanisms with post-hoc methods like saliency maps and SHAP values. The goal is to provide clear, visual, and robust explanations for model predictions in a financial context.

We focus on daily stock data (e.g., AAPL), enriched with technical indicators such as RSI, MACD, and stochastic oscillators. A custom attention-based LSTM (VixLSTM) is trained and evaluated on prediction accuracy and explanation quality.

---

## 📈 First Results

Our interpretable LSTM model (VixLSTM) achieves the following performance on the test set:

- **RMSE**: 0.0727  
- **MAPE**: 6.79%

These results show that the model effectively captures temporal dependencies and outperforms a naive baseline. Visual explanations show that the model focuses attention on recent price dips and key momentum indicators during volatile periods.
