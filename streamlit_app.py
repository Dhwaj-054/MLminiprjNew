"""Streamlit app for Mood-based Song Recommender

Run with:
    streamlit run streamlit_app.py

Features:
- Show model comparison table and simple charts (accuracy / F1 / training time)
- Button to (re)train models from CSV (runs `train_and_save_all`)
- Mood input (selectbox) for recommendations; shows top-5 songs
"""
import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import LabelEncoder

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.mood_recommender import train_and_save_all, recommend_songs

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
    # Load dataset and give clearer error if not found
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Make sure `data_moods.csv` is in the project root: {ROOT}")
    return pd.read_csv(DATA_PATH)


def show_model_comparison(results):
    # Build a DataFrame containing accuracy, precision, recall, f1 for each model
    rows = []
    model_keys = ['random_forest', 'xgboost', 'lightgbm']
    name_map = {'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM'}
    for k in model_keys:
        v = results.get(k, {}) if results else {}
        available = bool(v) and ('accuracy' in v)
        rows.append({
            'model_key': k,
            'model': name_map.get(k, k),
            'accuracy': float(v.get('accuracy', 0.0)) if available else 0.0,
            'precision': float(v.get('precision', 0.0)) if available else 0.0,
            'recall': float(v.get('recall', 0.0)) if available else 0.0,
            'f1': float(v.get('f1', 0.0)) if available else 0.0,
            'available': available,
        })
    dfm = pd.DataFrame(rows)

    # Melt to long format for grouped bar chart
    dfm_melt = dfm.melt(id_vars=['model_key', 'model'], value_vars=['accuracy', 'precision', 'recall', 'f1'], var_name='metric', value_name='score')

    # Choose best model by accuracy then f1
    best_row = dfm.sort_values(['accuracy', 'f1'], ascending=[False, False]).iloc[0]
    best_key = best_row['model_key']
    best_name = best_row['model']

    # Grouped bar chart using Altair: metrics on x, model as color
    chart = alt.Chart(dfm_melt).mark_bar(size=30).encode(
        x=alt.X('metric:N', title='Metric'),
        y=alt.Y('score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('model:N', title='Model'),
        column=alt.Column('model:N', header=alt.Header(labelAngle=0)),
        tooltip=['model', 'metric', alt.Tooltip('score', format='.3f')]
    ).properties(height=260)

    st.subheader('Model Performance Comparison')
    st.altair_chart(chart, use_container_width=True)

    # Simple accuracy-only chart as requested: show accuracies with models on x-axis
    df_acc = dfm[['model', 'accuracy']].copy()
    df_acc['is_best'] = df_acc['model'] == best_name
    acc_chart = alt.Chart(df_acc).mark_bar(size=50).encode(
        x=alt.X('model:N', title='Model'),
        y=alt.Y('accuracy:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(alt.datum.is_best, alt.value('#16a34a'), alt.value('#3b82f6')),
        tooltip=[alt.Tooltip('model:N'), alt.Tooltip('accuracy:Q', format='.3f')]
    ).properties(title='Accuracies', height=320)

    st.altair_chart(acc_chart, use_container_width=True)

    st.markdown(f"**Best model selected:** **{best_name}** — accuracy **{best_row['accuracy']:.3f}**, f1 **{best_row['f1']:.3f}**")

    # Show numeric results table
    st.dataframe(dfm.set_index('model')[['accuracy', 'precision', 'recall', 'f1']].style.format({
        'accuracy': '{:.3f}', 'precision': '{:.3f}', 'recall': '{:.3f}', 'f1': '{:.3f}'
    }))


def main():
    st.set_page_config(
        page_title="Mood-based Song Recommender",
        page_icon="🎵",
        layout="wide"
    )
    
    st.markdown(
        """
        <div style="padding:12px; background:#0b1220; border-radius:8px;">
            <h3 style="color:#9ca3af; margin:0; font-size:14px;">Machine Learning Mini-Project</h3>
            <h1 style="color:#fff; margin:6px 0 0 0; font-size:34px;">Mood based Music Recommender</h1>
            <p style="color:#cbd5e1; margin:6px 0 0 0; font-size:13px;">Select a mood and get personalized song recommendations based on audio features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for actions
    st.sidebar.header('Model Training')
    if st.sidebar.button('(Re)train Models', help='Train/retrain all models on the dataset'):
        with st.spinner('Training models — this may take several minutes...'):
            try:
                results, best, df_full, le, features = train_and_save_all()
                st.success(f'Training completed! Best model: {best["name"]} (accuracy: {best["accuracy"]:.3f})')
                # clear caches to reload saved artifacts
                load_results.clear()
                load_model_bundle.clear()
            except Exception as e:
                st.error(f'Training failed: {str(e)}')

    # Load saved results and model
    results = load_results()
    bundle = load_model_bundle()

    if results is None:
        st.warning('⚠️ No trained models found. Click "(Re)train Models" in the sidebar to train the models.')
    else:
        st.header('📊 Model Comparison')
        show_model_comparison(results)
        # compute best model (by accuracy, tie-breaker f1)
        best_name = None
        best_acc = -1.0
        best_f1 = -1.0
        for k, v in results.items():
            acc = v.get('accuracy', 0)
            f1 = v.get('f1', 0)
            if (acc > best_acc) or (acc == best_acc and f1 > best_f1):
                best_acc = acc
                best_f1 = f1
                best_name = k
        if best_name is not None:
            st.markdown(f"**Best model (selected):** **{best_name}** — accuracy: **{best_acc:.3f}**, f1: **{best_f1:.3f}**")

    # Recommendation section
    st.header('🎼 Get Song Recommendations')
    df = load_data()

    # Get mood classes from saved bundle or dataset
    classes = None
    if bundle is not None and 'label_encoder' in bundle:
        le = bundle['label_encoder']
        classes = list(le.classes_)
    else:
        if 'mood' in df.columns:
            classes = sorted(df['mood'].dropna().unique().tolist())

    col1, col2 = st.columns([2, 3])
    
    with col1:
        mood_input = None
        if classes:
            mood_input = st.selectbox('Choose a mood:', ['-- Select mood --'] + classes)
        else:
            mood_input = st.text_input('Enter mood:')

    with col2:
        if st.button('Get Recommendations 🎵', help='Click to get song recommendations for selected mood'):
            if mood_input is None or mood_input == '-- Select mood --' or str(mood_input).strip() == '':
                st.warning('Please choose a mood first!')
            else:
                try:
                    # prepare label encoder
                    if bundle is not None and 'label_encoder' in bundle:
                        le = bundle['label_encoder']
                    else:
                        le = LabelEncoder()
                        le.fit(df['mood'].astype(str))

                    # ensure mood_label column exists
                    if 'mood_label' not in df.columns:
                        df['mood_label'] = le.transform(df['mood'].astype(str))

                    features = bundle['features'] if bundle is not None and 'features' in bundle else [
                        c for c in ['tempo', 'energy', 'valence', 'danceability'] if c in df.columns
                    ]

                    recs = recommend_songs(df, mood_input, le, features, top_k=5)
                    if recs is None or recs.empty:
                        st.info(f'No recommendations found for mood "{mood_input}"')
                    else:
                        st.subheader(f'🎧 Top recommendations for "{mood_input}"')
                        # Style the recommendations table
                        st.dataframe(
                            recs.style.highlight_max(axis=0, props='background-color: #c6e6de'),
                            height=200
                        )
                except Exception as e:
                    st.error(f'Error getting recommendations: {str(e)}')

    # Footer with creators
    st.markdown("---")
    st.markdown(
        "<div style='color:#94a3b8; font-size:12px;'>Created by Dhwaj, Chaitanya, Dev, Dev</div>",
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()