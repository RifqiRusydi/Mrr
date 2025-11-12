import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

#Load Dataset

data={

    'Coffee Type':['Malaysiano','Espresso','Indiano','Mocha','Cappacuno','Latte'],
    'Sales':[300,500,100,900,72,1827]
}

df=pd.DataFrame(data)
best_sale=df[df['Sales']==df['Sales'].max()]
worst_sale=df[df['Sales']==df['Sales'].min()]

st.title("International Coffee Garden")
st.subheader("Bar Chart for Coffee Sales")
fig, ax = plt.subplots()
ax.bar(df['Coffee Type'], df['Sales'], color='skyblue')
ax.bar(best_sale['Coffee Type'], best_sale['Sales'], color='green', label='Best Sale')
ax.bar(worst_sale['Coffee Type'], worst_sale['Sales'], color='red', label='Worst Sale')
ax.legend()

st.pyplot(fig)

st.subheader("Pie Chart for Coffee Sales")
fig3, ax3 = plt.subplots()
colors=["#ff9999","#66b3ff","#99ff99","#ffcc99","#c2c2f0","#ffb3e6"]
ax3.pie(df['Sales'], labels=df['Coffee Type'], autopct='%1.1f%%', startangle=140, colors=colors)
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
st.pyplot(fig3)

st.write("Best Sale Coffee Type")
st.write(best_sale)
st.write("Worst Sale Coffee Type")
st.write(worst_sale)

print(best_sale)
print(worst_sale)