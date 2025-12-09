# Predicting Stock Price Movements from Social Media Sentiment

[![MIT License][license-shield]][license-url]


## About

Social media platforms like Twitter have become influential sources of retail investor sentiment, raising the question: can we predict short-term stock price movements by analyzing social media discussions? This project frames the problem as binary classification: given all social media posts about a stock on a trading day, predict whether the stock price will increase or decrease the next day.

We compare classical machine learning baselines (Logistic Regression on TF-IDF features) against deep learning approaches (MLP, BERT with classification head) on 80,793 tweets about 11 stocks from September 2021 to September 2022. Methods are evaluated using time-based splits to avoid lookahead bias and assessed via accuracy, F1 score, and AUC-ROC.

Results demonstrate the fundamental difficulty of this prediction task. All models barely exceeded random guessing (AUC-ROC near 0.500), with even the naive baseline (always predicting "down") achieving 61.6% accuracy by exploiting class imbalance. These negative results provide valuable evidence that social media sentiment alone is insufficient for short-term stock prediction given low signal-to-noise ratios, regime shifts, and market complexity.

Detailed findings are available in the [final report](final_report.pdf).

## Repository Structure
```
deep-learning-stock-sentiment/
├── data_raw/                  # Original Kaggle dataset
│   └── stock_tweets.csv
├── data_clean/                # Processed datasets and splits
│   ├── merged_dataset.parquet
│   ├── split_indices.npz
│   └── split_dates.json
├── outputs/                   # Evaluation utilities and results
│   ├── eval_utils.py
│   ├── tfidf_vectorizer.pkl
│   └── baseline_results.json
└── notebooks/                 # Implementation notebooks
    ├── 01_data_processing.ipynb
    ├── 02_mlp_baseline.ipynb
    └── 03_bert_model.ipynb
```

## Built With

* **Python** - Primary implementation language
* **TensorFlow/Keras** - Deep learning framework
* **scikit-learn** - Classical ML baselines and evaluation
* **pandas/numpy** - Data processing
* **BERT (keras-hub)** - Pre-trained language model

## Getting Started

Implementation is provided in three notebooks separating data processing, baseline models, and BERT experiments for independent reproducibility:

1. **01_data_processing.ipynb** - Data cleaning, aggregation, label creation, time-based splitting, TF-IDF vectorization, and Logistic Regression baseline
2. **02_mlp_baseline.ipynb** - MLP architecture with hyperparameter tuning on TF-IDF features
3. **03_bert_model.ipynb** - BERT feature extraction with MLP classification head

Notebooks are designed to run on Google Colab with GPU acceleration.

## Key Findings

- **Time-based splitting is critical**: Random splits leak future information; temporal splits reveal regime shifts (48% positive labels in training vs 38% in test)
- **All models perform poorly**: Best model achieves 0.544 AUC-ROC, barely above random chance (0.500)
- **Representation learning provides no advantage**: BERT (0.478 AUC) performs worse than TF-IDF baselines (0.544 AUC)
- **Complexity hurts without better representations**: MLP on TF-IDF underperforms Logistic Regression, indicating overfitting

## Team Members

|Name     |  Email   | 
|---------|-----------------|
|Nicholas Wong | nicwjh@mit.edu |
|Jean Bourseau | jeanb216@mit.edu |
|Lino Valette | linoval@mit.edu |
|Consti Casper | consti99@mit.edu |

## Course

15.776 Intensive Hands-on Deep Learning - MIT Sloan School of Management

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

Dataset: [Stock Tweets for Sentiment Analysis and Prediction](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction) by Yukhymenko (2022)

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://opensource.org/licenses/MIT
