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

def preprocess_data(df):
    # Преобразуем числовые признаки
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Удаляем категориальные признаки
    df = df.select_dtypes(include=['number'])
    
    # Разделяем данные по классам
    target_col = 39  # Замените на имя целевого столбца
    pos_class = df[df[target_col] == 1]
    neg_class = df[df[target_col] == 0]
    
    # Заполняем NaN средними значениями внутри классов
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df.loc[df[target_col] == 1, col] = pos_class[col].mean()
            df.loc[df[target_col] == 0, col] = neg_class[col].mean()
    
    return df

df = preprocess_data(df)
df = df.dropna(axis=1, how='any')


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
