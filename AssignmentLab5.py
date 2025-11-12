import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#from pandas.api.types import parallel_cordinates

df=pd.read_csv('Paintball.csv')
st.write(df.head(5))

st.title("High-Dimensional Data Visualization Techniques Lab")

with st.expander("Mosai Plot (Marimekko)"):
    st.write("A **mosaic Plot**")
    fig_mekko = px.histogram(df, x="Date",y="Equipment_Rental_Cost", color="Team", barmode="stack", histfunc="sum",height=500)
    fig_mekko.update_layout(yaxis_title="Equipment_Rental_Cost")
    st.plotly_chart(fig_mekko)

with st.expander("Trellis Display"):
    st.write("**trellis Display**")

    trellis_chart = alt.Chart(df).mark_point().encode(
        x='Date',y='Equipment_Rental_Cost',color='Team',facet=alt.Facet('Team:N', columns=4
                                                           ))
    st.altair_chart(trellis_chart, use_container_width=True)

with st.expander("HeatMap"):
    st.write("A **HeatMap Visualization**")
    pivot_table = df.pivot_table(values='Equipment_Rental_Cost', index='Date', columns='Team', aggfunc='mean')
    fig, ax = plt.subplots()
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)
with st.expander("Multivariate Scatter Plot"):
    st.write("A **Multivariate Scatter Plot**")
    scatter_plot= alt.Chart(df).mark_circle(size=60).encode(x='Equipment_Rental_Cost',y='Team',color='Team',size='Shots_Fired',tooltip=['Date','Equipment_Rental_Cost','Team','Hits','Win','Shots_Fired']).interactive()
    st.altair_chart(scatter_plot,use_container_width=True)

with st.expander("3D Scatter Plot"):
    st.write("A **3D Scatter Plot**")
    scatter_3d = px.scatter_3d(df, x='Equipment_Rental_Cost', y='Team', z='Date', color='Win', size='Shots_Fired', title='3D Scatter Plot of Tips Dataset')
    st.plotly_chart(scatter_3d) 

with st.expander("Parallel Coordinates Plot"):
    st.write("A **Parallel Coordinates Plot**")
    parallel_plot = px.parallel_coordinates(df, color='Shots_Fired', dimensions=['Equipment_Rental_Cost', 'Hits', 'Shots_Fired','Date'], color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=df['Shots_Fired'].mean())
    st.plotly_chart(parallel_plot)

with st.expander("box Plot"):
    st.write("A **Box Plot**")
    box_plot = px.box(df, x='Team', y='Equipment_Rental_Cost', color='Win', title='Box Plot of Equipment Rental Cost by Team and Win Status')
    st.plotly_chart(box_plot)

with st.expander("Violin Plot"):
    st.write("A **Violin Plot**")
    violin_plot = px.violin(df, x='Team', y='Equipment_Rental_Cost', color='Win', box=True, points='all', title='Violin Plot of Equipment Rental Cost by Team and Win Status')
    st.plotly_chart(violin_plot)

