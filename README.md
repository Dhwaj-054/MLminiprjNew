# Mood-based Song Recommender (Spotify Mood dataset)

This mini-project trains classifiers to predict mood labels from Spotify audio features and provides a simple recommendation function to return top songs matching a mood.

Contents
- `src/mood_recommender.py` — main module: EDA, preprocessing, training, evaluation, recommendation functions
- `scripts/smoke_test.py` — small script to verify data load
- `requirements.txt` — Python dependencies

Notes
- Place `data_moods.csv` in the project root (already present in the workspace).
- Install dependencies in a virtualenv: `python -m pip install -r requirements.txt`
- Run the main module to train models and start the interactive recommender: `python src/mood_recommender.py`
