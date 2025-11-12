import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#from pandas.api.types import parallel_cordinates

df=pd.read_csv('tips.csv')
st.write(df.head(5))

st.title("Lab 5: Data Visualization with Streamlit")

with st.expander("Mosai Plot (Marimekko)"):
    st.write("A **mosaic Plot**")
    fig_mekko = px.histogram(df, x="day",y="total_bill", color="sex", barmode="stack", histfunc="sum",height=500)
    fig_mekko.update_layout(yaxis_title="Total Bill")
    st.plotly_chart(fig_mekko)

with st.expander("Trellis Display"):
    st.write("**trellis Display**")

    trellis_chart = alt.Chart(df).mark_point().encode(
        x='total_bill',y='tip',color='sex',facet=alt.Facet('time:N', columns=4
                                                           ))
    st.altair_chart(trellis_chart, use_container_width=True)

with st.expander("HeatMap"):
    st.write("A **HeatMap Visualization**")
    pivot_table = df.pivot_table(values='total_bill', index='day', columns='time', aggfunc='mean')
    fig, ax = plt.subplots()
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)
with st.expander("Multivariate Scatter Plot"):
    st.write("A **Multivariate Scatter Plot**")
    scatter_plot= alt.Chart(df).mark_circle(size=60).encode(x='total_bill',y='tip',color='sex',size='size',tooltip=['total_bill','tip','sex','size','smoker']).interactive()
    st.altair_chart(scatter_plot,use_container_width=True)

with st.expander("Parallel Coordinates Plot"):
    st.write("A **Parallel Coordinates Plot**")
    parallel_plot = px.parallel_coordinates(df, color='size', dimensions=['total_bill', 'tip', 'size','day'], color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=df['size'].mean())
    st.plotly_chart(parallel_plot)

with st.expander("3D Scatter Plot"):
    st.write("A **3D Scatter Plot**")
    scatter_3d = px.scatter_3d(df, x='total_bill', y='tip', z='size', color='sex', symbol='smoker', size='total_bill', title='3D Scatter Plot of Tips Dataset')
    st.plotly_chart(scatter_3d) 
