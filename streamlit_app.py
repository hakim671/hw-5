import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
  options = [0, 1, 3, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
  col = st.multiselect("Выберите колоки для отображения", options)
  st.write("X")
  X_row = df.drop(39, axis=1)
  st.dataframe(X_row[col])
  st.write("y")
  y_row = df[39]
  st.dataframe(y_row)
with st.sidebar:
    st.subheader("Признаки")
    f0 = st.slider('Feature-0', float(np.min(df[0])), float(np.max(df[0])), float(np.std(df[0])))
    f1 = st.slider('Feature-1', float(np.min(df[1])), float(np.max(df[1])), float(np.std(df[1])))
    f3 = st.slider('Feature-3', float(np.min(df[3])), float(np.max(df[3])), float(np.std(df[3])))
    f15 = st.slider('Feature-15', float(np.min(df[15])), float(np.max(df[15])), float(np.std(df[15])))
    f16 = st.slider('Feature-16', float(np.min(df[16])), float(np.max(df[16])), float(np.std(df[16])))
    f19 = st.slider('Feature-19', float(np.min(df[19])), float(np.max(df[19])), float(np.std(df[19])))
    f20 = st.slider('Feature-20', float(np.min(df[20])), float(np.max(df[20])), float(np.std(df[20])))
    f21 = st.slider('Feature-21', float(np.min(df[21])), float(np.max(df[21])), float(np.std(df[21])))
    f22 = st.slider('Feature-22', float(np.min(df[22])), float(np.max(df[22])), float(np.std(df[22])))
    f23 = st.slider('Feature-23', float(np.min(df[23])), float(np.max(df[23])), float(np.std(df[23])))
    f24 = st.slider('Feature-24', float(np.min(df[24])), float(np.max(df[24])), float(np.std(df[24])))
    f25 = st.slider('Feature-25', float(np.min(df[25])), float(np.max(df[25])), float(np.std(df[25])))
    f26 = st.slider('Feature-26', float(np.min(df[26])), float(np.max(df[26])), float(np.std(df[26])))
    f27 = st.slider('Feature-27', float(np.min(df[27])), float(np.max(df[27])), float(np.std(df[27])))
    f28 = st.slider('Feature-28', float(np.min(df[28])), float(np.max(df[28])), float(np.std(df[28])))
    f29 = st.slider('Feature-29', float(np.min(df[29])), float(np.max(df[29])), float(np.std(df[29])))
    f30 = st.slider('Feature-30', float(np.min(df[30])), float(np.max(df[30])), float(np.std(df[30])))
    f31 = st.slider('Feature-31', float(np.min(df[31])), float(np.max(df[31])), float(np.std(df[31])))
    f32 = st.slider('Feature-32', float(np.min(df[32])), float(np.max(df[32])), float(np.std(df[32])))
    f33 = st.slider('Feature-33', float(np.min(df[33])), float(np.max(df[33])), float(np.std(df[33])))
    f34 = st.slider('Feature-34', float(np.min(df[34])), float(np.max(df[34])), float(np.std(df[34])))
    f35 = st.slider('Feature-35', float(np.min(df[35])), float(np.max(df[35])), float(np.std(df[35])))
    f36 = st.slider('Feature-36', float(np.min(df[36])), float(np.max(df[36])), float(np.std(df[36])))
    f37 = st.slider('Feature-37', float(np.min(df[37])), float(np.max(df[37])), float(np.std(df[37])))
    f38 = st.slider('Feature-38', float(np.min(df[38])), float(np.max(df[38])), float(np.std(df[38])))
    data = {'f0': f0,
            'f1': f1,
            'f3': f3,
            'f15': f15,
            'f16': f16,
            'f19': f19,
            'f20': f20,
            'f21': f21,
            'f22': f22,
            'f23': f23,
            'f24': f24,
            'f25': f25,
            'f26': f26,
            'f27': f27,
            'f28': f28,
            'f29': f29,
            'f30': f30,
            'f31': f31,
            'f32': f32,
            'f33': f33,
            'f34': f34,
            'f35': f35,
            'f36': f36,
            'f37': f37,
            'f38': f38}
    input_df = pd.DataFrame(data, index=[0])
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_row, y_row)
fig = px.histogram(
    df, 
    x=39, 
    nbins=10, 
    title='idk'
)
st.plotly_chart(fig)

if st.button("Predict"):
    st.write(rf.predict(input_df))
