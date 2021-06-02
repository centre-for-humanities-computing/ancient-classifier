import streamlit as st
import pandas as pd

@st.cache
def extract_decision_function(transformer, model, user_input):
    '''
    Returns a) assigned labels, b) melted df with prediction confidence per every label
    '''
    
    X = transformer.transform([user_input])

    y_pred = model.predict(X)
    y_pred_confidence = model.decision_function(X)

    confidence_df = pd.DataFrame(y_pred_confidence, columns=model.classes_)
    confidence_df = confidence_df.melt(value_name='confidence', var_name='type')

    return y_pred, confidence_df