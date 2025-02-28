import streamlit as st
import pandas as pd

df = pd.read_csv("bands.data", header=None)
df = df.dropna(subset=[39])

df[39] = df[39].replace({
    'band':1,
    'noband':0
})
df = df.replace({
    '?':None
})


with st.expander('data'):
  st.write("X")
  X_row = df.drop(39, axis=1)
  st.dataframe(X_row)
  st.write("y")
  y_row = df[39]
  st.dataframe(y_row)
with st.sidebar:
  st.header("Признаки")
  island = st.selectbox("Island", ("Torgersen", "Dream", "Biscoe"))
  bill_length_mm = st.slider('Bill length(mm)', 32.1, 59.6, 44.5)
  bill_depth_mm = st.slider("bill_depth_(mm)", 13.1, 21.5, 17.3)
  flipper_length_mm = st.slider("flipper_length_mm", 32.1, 59.6, 44.5)
  body_mass_g = st.slider("body_mass_g", 32.1, 59.6, 44.5)
  gender = st.selectbox("gender", ("female", "male"))
