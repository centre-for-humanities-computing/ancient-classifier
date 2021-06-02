import os
import pickle

import pandas as pd
import streamlit as st
import altair as alt

from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from plots import plot_bar_confidence
from util import extract_decision_function

# ===
# Externals
# ===

with open('mdl/conservative_oversampled/210603_ridge.pcl', 'rb') as fin:
    mdl_clf_conservative = pickle.load(fin)

with open('mdl/conservative_oversampled/210603_preprocessing.pcl', 'rb') as fin:
    mdl_prep_conservative = pickle.load(fin)

with open('mdl/interpretive_oversampled/210602_ridge.pcl', 'rb') as fin:
    mdl_clf_interpretive = pickle.load(fin)

with open('mdl/interpretive_oversampled/210602_preprocessing.pcl', 'rb') as fin:
    mdl_prep_interpretive = pickle.load(fin)

# ===
# Streamlit config
# ===
st.set_page_config(
    page_icon='🏛',
    layout='centered',
    page_title='Ancient Classifier'
)

# ===
# Sidebar
# ===
with st.sidebar:
    # track which model to use 
    model_choice = st.selectbox(
        'Which model do you want to use?',
        ('As-is on inscriptions (conservative text)', 'Full sentences (interpretive text)')
    )

    if model_choice == 'As-is on inscriptions (conservative text)':
        default_input = 'Accae l Myrine Accae l Sympherusae M Ant M l Ero poma'

    elif model_choice == 'Full sentences (interpretive text)':
        default_input = 'Accae mulieris libertae Myrine Accae mulieris libertae Sympherusae Marco Antonio Marci liberto Ero pomario'


# ===
# Classify input
# ===
st.title('Ancient Classifier')
st.write('Label ancient Mediterranean roadside inscriptions. [Source code](https://github.com/centre-for-humanities-computing/ancient-classifier/), forked from the [Epigraphic Roads](https://github.com/sdam-au/epigraphic_roads/) project.')

user_input = st.text_area("Input text", default_input)

if not isinstance(user_input, str):
    raise TypeError('Input not string')

if len(user_input) > 1000:
    raise MemoryError('Input text too long. Max 1000 characters allowed')

if st.button('Classify!'):

    if model_choice == 'As-is on inscriptions (conservative text)':
        transformer = mdl_prep_conservative
        model = mdl_clf_conservative
    elif model_choice == 'Full sentences (interpretive text)':
        transformer = mdl_prep_interpretive
        model = mdl_clf_interpretive

    y_pred, confidence_df = extract_decision_function(
        transformer,
        model,
        user_input
    )

    st.write('\n\n')
    st.markdown("""---""")

    st.subheader('Text was classified as')
    st.code(f'{y_pred}')
    st.write('\n\n')

    st.subheader('Confidence')
    st.write('\n')
    st.write(
        plot_bar_confidence(confidence_df)
    )


# ===
# More info about the model
# ===
