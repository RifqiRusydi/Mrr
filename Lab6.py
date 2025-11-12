import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static

st.set_page_config(layout="wide")

st.sidebar.title("Change Map color")
color_option = st.sidebar.selectbox("Select Map Color Theme", ["YlOrRd", "BuGn"])
if color_option == "YlOrRd":
    color="YlOrRd"
if color_option == "BuGn":
    color="BuGn"

def generate_population_data():
    states = [
        "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", 
        "Pahang", "Penang", "Perak", "Perlis", "Sabah", "Sarawak", 
        "Selangor", "Terengganu", "Kuala Lumpur", "Labuan", "Putrajaya"
    ]
    population = np.random.randint(500000, 5000000, size=len(states))
    data = pd.DataFrame({"State": states, "Population": population})
    return data

def generate_gdp_data():
    states = [
        "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", 
        "Pahang", "Penang", "Perak", "Perlis", "Sabah", "Sarawak", 
        "Selangor", "Terengganu", "Kuala Lumpur", "Labuan", "Putrajaya"
    ]
    population = np.random.randint(1000000, 10000000, size=len(states))
    gdp = np.random.randint(10000, 100000, size=len(states))
    data = pd.DataFrame({"State": states, "GDP": gdp})
    return data


@st.cache_data
def load_map():
    malaysia_map = gpd.read_file("malaysia_states.geojson")
    return malaysia_map

def plot_map(malaysia_map, data):
    m = folium.Map(location=[4.2105, 101.9758], zoom_start=6)
    folium.Choropleth(
        geo_data=malaysia_map,
        name="Population Distribution",
        data=data,
        columns=["State", "Population"],
        key_on="feature.properties.name",
        fill_color=color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Population Distribution",
    ).add_to(m)
    return m

def plot_gdp_map(malaysia_map, data):
    m = folium.Map(location=[4.2105, 101.9758], zoom_start=6)
    folium.Choropleth(
        geo_data=malaysia_map,
        name="GDP Distribution",
        data=data,
        columns=["State", "GDP"],
        key_on="feature.properties.name",
        fill_color=color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="GDP Distribution",
    ).add_to(m)
    return m

def main():
    st.title("Malaysia Population Distribution Map")
    population_data = generate_population_data()
    st.write("Sample Population Data for States:")
    st.dataframe(population_data)
    
    malaysia_map = load_map()
    folium_map = plot_map(malaysia_map, population_data)
    folium_static(folium_map)


def main_gdp():
    st.title("Malaysia GDP Distribution Map")
    gdp_data = generate_gdp_data()
    st.write("Sample GDP Data for States:")
    st.dataframe(gdp_data)
    
    malaysia_map = load_map()
    folium_map = plot_gdp_map(malaysia_map, gdp_data)
    folium_static(folium_map)




if __name__ == "__main__":
    main()
    st.markdown("---")
    main_gdp()