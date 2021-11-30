import streamlit as st
import pandas as pd

st.write("""
# My first application
Heelo *world!*
""")

df = pd.read_csv('/mnt/data/SHAVE_cases/Analysis/JTTI/practicum30mindata.csv')
st.line_chart(df['iMESH'])
