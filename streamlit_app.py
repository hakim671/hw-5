import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    st.subheader("Признаки")
    f0 = st.slider('Feature-0', int(np.min(df[0])), int(np.max(df[0])), int(np.mean(df[0])))
    f1 = st.slider('Feature-1', int(np.min(df[1])), int(np.max(df[1])), int(np.mean(df[1])))
    f3 = st.slider('Feature-3', int(np.min(df[3])), int(np.max(df[3])), int(np.mean(df[3])))
    f15 = st.slider('Feature-15', int(np.min(df[15])), int(np.max(df[15])), int(np.mean(df[15])))
    f16 = st.slider('Feature-16', int(np.min(df[16])), int(np.max(df[16])), int(np.mean(df[16])))
    f19 = st.slider('Feature-19', int(np.min(df[19])), int(np.max(df[19])), int(np.mean(df[19])))
    f20 = st.slider('Feature-20', int(np.min(df[20])), int(np.max(df[20])), int(np.mean(df[20])))
    f21 = st.slider('Feature-21', int(np.min(df[21])), int(np.max(df[21])), int(np.mean(df[21])))
    f22 = st.slider('Feature-22', int(np.min(df[22])), int(np.max(df[22])), int(np.mean(df[22])))
    f23 = st.slider('Feature-23', int(np.min(df[23])), int(np.max(df[23])), int(np.mean(df[23])))
    f24 = st.slider('Feature-24', int(np.min(df[24])), int(np.max(df[24])), int(np.mean(df[24])))
    f25 = st.slider('Feature-25', int(np.min(df[25])), int(np.max(df[25])), int(np.mean(df[25])))
    f26 = st.slider('Feature-26', int(np.min(df[26])), int(np.max(df[26])), int(np.mean(df[26])))
    f27 = st.slider('Feature-27', int(np.min(df[27])), int(np.max(df[27])), int(np.mean(df[27])))
    f28 = st.slider('Feature-28', int(np.min(df[28])), int(np.max(df[28])), int(np.mean(df[28])))
    f29 = st.slider('Feature-29', int(np.min(df[29])), int(np.max(df[29])), int(np.mean(df[29])))
    f30 = st.slider('Feature-30', int(np.min(df[30])), int(np.max(df[30])), int(np.mean(df[30])))
    f31 = st.slider('Feature-31', int(np.min(df[31])), int(np.max(df[31])), int(np.mean(df[31])))
    f32 = st.slider('Feature-32', int(np.min(df[32])), int(np.max(df[32])), int(np.mean(df[32])))
data = {'f0':f0,
        'f1':f1,
        'f3':f3,
        'f15':f15,
        'f16':f16,
        'f19':f19,
        'f20':f20,
        'f21':f21,
        'f22':f22,
        'f23':f23,
        'f24':f24,
        'f25':f25,
        'f26':f26,
        'f27':f27,
        'f28':f28,
        'f29':f29,
        'f30':f30,
        'f31':f31,
        'f32':f32}
