# GameSense AI

A Valorant esports analytics project for identifying potential player performance deterioration using statistical indicators, player-adaptive baselines, and machine learning.

## Dataset

Match and player statistics are sourced from [VCT Reference](https://vct-reference.com/dataset).

The dataset is free to use, including commercially, subject to the VCT Reference Terms of Service.

Source: https://vct-reference.com/dataset

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Scikit-learn
- SciPy
- TF-IDF and cosine similarity

## Run Locally

1. Clone this repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Start the dashboard:

   streamlit run dashboard/app.py

## Project Structure

- dashboard/ — Streamlit dashboard
- chatbot/ — Offline NLP chatbot
- data/features/ — Processed analytical datasets
- scripts/ — Data processing, feature engineering, and model evaluation
- models/ — Saved machine learning models (excluded from the initial repository)

## Disclaimer

GameSense AI provides statistical performance analysis and early-warning signals. It does not diagnose a player's mental or psychological state.
