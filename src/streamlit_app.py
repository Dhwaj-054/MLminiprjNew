"""Streamlit app for Mood-based Song Recommender

Run with:
    streamlit run src/streamlit_app.py

Features:
- Show model comparison table and simple charts (accuracy / F1 / training time)
- Button to (re)train models from CSV (runs `train_and_save_all`)
- Mood input (selectbox) for recommendations; shows top-5 songs
"""
import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Import the mood_recommender module from current directory
import mood_recommender as mr

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
DATA_PATH = ROOT / 'data_moods.csv'


@st.cache_data
def load_results():
    p = MODEL_DIR / 'results.joblib'
    if p.exists():
        return joblib.load(p)
    return None


@st.cache_data
def load_model_bundle():
    p = MODEL_DIR / 'best_model.joblib'
    if p.exists():
        return joblib.load(p)
    return None


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def show_model_comparison(results):
    # Build a DataFrame for display
    rows = []
    for k, v in results.items():
        rows.append({
            'model': k,
            'accuracy': v.get('accuracy', 0),
            'f1': v.get('f1', 0),
            'train_time_s': v.get('train_time', 0),
        })
    df = pd.DataFrame(rows).set_index('model')
    st.dataframe(df)

    # Charts
    st.subheader('Accuracy vs F1')
    st.bar_chart(df[['accuracy', 'f1']])

    st.subheader('Training times (s)')
    st.bar_chart(df['train_time_s'])


def main():
    st.title('Mood-based Song Recommender')

    st.sidebar.header('Actions')
    if st.sidebar.button('(Re)train models'):
        with st.spinner('Training models — this may take several minutes...'):
            results, best, df_full, le, features = mr.train_and_save_all()
            st.success('Training completed')
            # clear caches to reload saved artifacts
            load_results.clear()
            load_model_bundle.clear()

    results = load_results()
    bundle = load_model_bundle()

    if results is None:
        st.info('No trained results found. Use the (Re)train models button to train from CSV.')
    else:
        st.header('Model comparison')
        show_model_comparison(results)

    st.header('Recommend songs by mood')
    df = load_data()

    # Get mood classes from saved bundle or dataset
    classes = None
    if bundle is not None and 'label_encoder' in bundle:
        le = bundle['label_encoder']
        classes = list(le.classes_)
    else:
        if 'mood' in df.columns:
            classes = sorted(df['mood'].dropna().unique().tolist())

    mood_input = None
    if classes:
        mood_input = st.selectbox('Choose a mood', ['-- pick --'] + classes)
    else:
        mood_input = st.text_input('Enter mood')

    if st.button('Recommend'):
        if mood_input is None or mood_input == '-- pick --' or str(mood_input).strip() == '':
            st.warning('Please choose or enter a mood')
        else:
            try:
                # prepare label encoder
                if bundle is not None and 'label_encoder' in bundle:
                    le = bundle['label_encoder']
                else:
                    # create a temporary label encoder from dataset
                    le = mr.LabelEncoder()
                    le.fit(df['mood'].astype(str))

                # ensure mood_label column exists
                if 'mood_label' not in df.columns:
                    df['mood_label'] = le.transform(df['mood'].astype(str))

                features = bundle['features'] if bundle is not None and 'features' in bundle else [
                    c for c in ['tempo', 'energy', 'valence', 'danceability'] if c in df.columns
                ]

                recs = mr.recommend_songs(df, mood_input, le, features, top_k=5)
                if recs is None or recs.empty:
                    st.info(f'No recommendations found for mood "{mood_input}"')
                else:
                    st.subheader(f'Top recommendations for "{mood_input}"')
                    st.table(recs)
            except Exception as e:
                st.error(str(e))


if __name__ == '__main__':
    main()