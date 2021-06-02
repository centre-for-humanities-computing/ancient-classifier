import os
import pickle

import pandas as pd
import streamlit as st
import altair as alt

from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from plots import plot_bar_confidence

# ===
# Externals
# ===
with open('../mdl/210602_ridge.pcl', 'rb') as fin:
    classifier = pickle.load(fin)

with open('../mdl/210602_preprocessing.pcl', 'rb') as fin:
    preprocessing = pickle.load(fin)

labels = classifier.classes_

# ===
# Streamlit config
# ===
st.set_page_config(
    page_icon='🏛',
    layout='centered',
    page_title='Ancient Classifier'
)

# ===
# Classify input
# ===
st.title('Ancient Classifier')
st.write('Label ancient Mediterranean roadside inscriptions. [Source code](https://github.com/centre-for-humanities-computing/ancient-classifier/), forked from the [Epigraphic Roads](https://github.com/sdam-au/epigraphic_roads/) project.')

default_input = 'Accae l Myrine Accae l Sympherusae M Ant M l Ero poma'
user_input = st.text_area("Input conservative text", default_input)

if not isinstance(user_input, str):
    raise TypeError('Input not string')


if st.button('Classify!'):

    X = preprocessing.transform([user_input])

    y_pred = classifier.predict(X)
    y_pred_confidence = classifier.decision_function(X)

    confidence_df = pd.DataFrame(y_pred_confidence, columns=labels)
    confidence_df = confidence_df.melt(value_name='confidence', var_name='label')

    st.write(f'Instription was classified as {y_pred}')
    st.write(
        plot_bar_confidence(confidence_df)
    )


# ===
# More info about the model
# ===
