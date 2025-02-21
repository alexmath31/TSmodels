import streamlit as st

def select_feature(df):
    if "selected_feature" not in st.session_state:
        st.session_state.selected_feature = None
    
    st.write("### Select a Feature for Forecasting:")
    cols = df.columns.tolist()

    col_buttons = st.columns(len(cols))

    for i, col in enumerate(cols):
        if col_buttons[i].button(col):
            st.session_state.selected_feature = col
    
    return st.session_state.selected_feature
