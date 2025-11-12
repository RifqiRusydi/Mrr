import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

#Load Dataset

data={

    'Month':['Jan','Feb','March','April','June'],
    'Sales':[3000,2500,3800,3850,4100]
}
df=pd.DataFrame(data)

Best_Month=df[df["Sales"] == df["Sales"].max()].index[0]
Worst_Month=df['Sales'].idxmin()

st.title("Monthly Sales for Paintball")
fig, ax=plt.subplots()

ax.plot(df['Month'], df['Sales'], linestyle='-', color='black')
ax.plot(df['Month'][Best_Month], df['Sales'][Best_Month], marker='o', color='green', label='Best Month')
ax.plot(df['Month'][Worst_Month], df['Sales'][Worst_Month], marker='o', color='red', label='Worst Month')
ax.legend()

st.pyplot(fig)
