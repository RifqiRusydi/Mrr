#22009429
#Assignment 3
#SIP
#Seri Iskandar Paintball

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


data_sales = {
    'Month': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    'Markers': [1200,1350,1600,1500,1700,1800,1900,2000,2100,1950,1850,2100],
    'Masks': [800,950,900,1000,1100,1150,1200,1250,1300,1400,1350,1450],
    'Pellets': [3000,2800,3200,3500,3700,3900,4100,4300,4400,4600,4800,4900]
}


df_sales= pd.DataFrame(data_sales)

best_month_markers_sales=df_sales['Markers'].idxmax()
best_month_masks_sales=df_sales['Masks'].idxmax()
best_month_pellets_sales=df_sales['Pellets'].idxmax()

worst_month_markers_sales=df_sales['Markers'].idxmin()
worst_month_masks_sales=df_sales['Masks'].idxmin()
worst_month_pellets_sales=df_sales['Pellets'].idxmin()


st.sidebar.title("Navigation")
page = st.sidebar.selectbox('Choose Page', ["Home","Sales Data"])

if page == "Home":
    st.title("Welcome to Seri Iskandar Paintball Center")
    st.subheader("Choose page from Navigation to explore the monthly sales.")

elif page == "Sales Data":

    st.sidebar.subheader("Select Product for Sales Data")
    product = st.sidebar.selectbox('Choose Product', ["Markers", "Masks", "Pellets","All Products"])
    st.title("Monthly Sales Data for Paintball Products")

    if product == "Markers":

        st.sidebar.subheader("Select Type of Data Visualization")
        chart_type = st.sidebar.selectbox('Choose Chart Type', ["Bar Chart", "Pie Chart"])
        st.subheader("Monthly Sales Data for Paintball Markers")

        if chart_type == "Bar Chart":

            st.markdown("Bar Chart for Paintball Markers Sales")
            if st.checkbox("Show The Best and Worst Month", value=False):
                fig, ax=plt.subplots()
                
                ax.bar(df_sales['Month'][best_month_markers_sales], df_sales['Markers'][best_month_markers_sales], color='green', label='Best Month')
                ax.bar(df_sales['Month'][worst_month_markers_sales], df_sales['Markers'][worst_month_markers_sales], color='red', label='Worst Month')
                ax.legend()
                st.pyplot(fig)

            else:

                sales_min_filter=st.slider("Sales More Than", min_value=int(df_sales['Markers'].min()), max_value=int(df_sales['Markers'].max()), value=int(df_sales['Markers'].min()), step=100)
                sales_max_filter=st.slider("Sales Less Than", min_value=int(df_sales['Markers'].min()), max_value=int(df_sales['Markers'].max()), value=int(df_sales['Markers'].max()), step=100)
                filtered_sale=df_sales[(df_sales['Markers'] >= sales_min_filter) & (df_sales['Markers'] <= sales_max_filter)]
                fig, ax=plt.subplots()
                ax.bar(filtered_sale['Month'], filtered_sale['Markers'], color='skyblue')
                st.pyplot(fig)


        elif chart_type == "Pie Chart":


            st.markdown("Pie Chart for Paintball Markers Sales")
            fig2, ax2=plt.subplots()
            colors=["#ff9999","#66b3ff","#99ff99","#ffcc99","#c2c2f0","#ffb3e6","#c2f0c2","#f0b3ff","#ff6666","#66ffe0","#ffb366","#b3b3b3"]
            ax2.pie(df_sales['Markers'], labels=df_sales['Month'], autopct='%1.1f%%', startangle=140, colors=colors)
            ax2.axis('equal') 
            st.pyplot(fig2)


    elif product == "Masks":

        st.sidebar.subheader("Select Type of Data Visualization")
        chart_type = st.sidebar.selectbox('Choose Chart Type', ["Bar Chart", "Pie Chart"])
        st.subheader("Monthly Sales Data for Paintball Masks")

        if chart_type == "Bar Chart":

            st.markdown("Bar Chart for Paintball Masks Sales")
            if st.checkbox("Show The Best and Worst Month", value=False):
                fig, ax=plt.subplots()
                
                ax.bar(df_sales['Month'][best_month_masks_sales], df_sales['Masks'][best_month_masks_sales], color='green', label='Best Month')
                ax.bar(df_sales['Month'][worst_month_masks_sales], df_sales['Masks'][worst_month_masks_sales], color='red', label='Worst Month')
                ax.legend()
                st.pyplot(fig)

            else:

                sales_min_filter=st.slider("Sales More Than", min_value=int(df_sales['Masks'].min()), max_value=int(df_sales['Masks'].max()), value=int(df_sales['Masks'].min()), step=100)
                sales_max_filter=st.slider("Sales Less Than", min_value=int(df_sales['Masks'].min()), max_value=int(df_sales['Masks'].max()), value=int(df_sales['Masks'].max()), step=100)
                filtered_sale=df_sales[(df_sales['Masks'] >= sales_min_filter) & (df_sales['Masks'] <= sales_max_filter)]
                fig, ax=plt.subplots()
                ax.bar(filtered_sale['Month'], filtered_sale['Masks'], color='skyblue')
                st.pyplot(fig)


        elif chart_type == "Pie Chart":


            st.markdown("Pie Chart for Paintball Masks Sales")
            fig2, ax2=plt.subplots()
            colors=["#ff9999","#66b3ff","#99ff99","#ffcc99","#c2c2f0","#ffb3e6","#c2f0c2","#f0b3ff","#ff6666","#66ffe0","#ffb366","#b3b3b3"]
            ax2.pie(df_sales['Masks'], labels=df_sales['Month'], autopct='%1.1f%%', startangle=140, colors=colors)
            ax2.axis('equal') 
            st.pyplot(fig2)


    elif product == "Pellets":
        st.sidebar.subheader("Select Type of Data Visualization")
        chart_type = st.sidebar.selectbox('Choose Chart Type', ["Bar Chart", "Pie Chart"])
        st.subheader("Monthly Sales Data for Paintball Pellets")

        if chart_type == "Bar Chart":

            st.markdown("Bar Chart for Paintball Pellets Sales")
            if st.checkbox("Show The Best and Worst Month", value=False):
                fig, ax=plt.subplots()
                
                ax.bar(df_sales['Month'][best_month_pellets_sales], df_sales['Pellets'][best_month_pellets_sales], color='green', label='Best Month')
                ax.bar(df_sales['Month'][worst_month_pellets_sales], df_sales['Pellets'][worst_month_pellets_sales], color='red', label='Worst Month')
                ax.legend()
                st.pyplot(fig)

            else:

                sales_min_filter=st.slider("Sales More Than", min_value=int(df_sales['Pellets'].min()), max_value=int(df_sales['Pellets'].max()), value=int(df_sales['Pellets'].min()), step=100)
                sales_max_filter=st.slider("Sales Less Than", min_value=int(df_sales['Pellets'].min()), max_value=int(df_sales['Pellets'].max()), value=int(df_sales['Pellets'].max()), step=100)
                filtered_sale=df_sales[(df_sales['Pellets'] >= sales_min_filter) & (df_sales['Pellets'] <= sales_max_filter)]
                fig, ax=plt.subplots()
                ax.bar(filtered_sale['Month'], filtered_sale['Pellets'], color='skyblue')
                st.pyplot(fig)


        elif chart_type == "Pie Chart":


            st.markdown("Pie Chart for Paintball Pellets Sales")
            fig2, ax2=plt.subplots()
            colors=["#ff9999","#66b3ff","#99ff99","#ffcc99","#c2c2f0","#ffb3e6","#c2f0c2","#f0b3ff","#ff6666","#66ffe0","#ffb366","#b3b3b3"]
            ax2.pie(df_sales['Pellets'], labels=df_sales['Month'], autopct='%1.1f%%', startangle=140, colors=colors)
            ax2.axis('equal') 
            st.pyplot(fig2)

    elif product == "All Products":
        st.subheader("Monthly Sales Data for All Paintball Products")
        st.markdown("Line Chart for All Paintball Products Sales")
        fig, ax=plt.subplots()
        ax.plot(df_sales['Month'], df_sales['Markers'], linestyle='-', color='blue', label='Markers')
        ax.plot(df_sales['Month'], df_sales['Masks'], linestyle='-',  color='orange', label='Masks')
        ax.plot(df_sales['Month'], df_sales['Pellets'], linestyle='-', color='green', label='Pellets')
        ax.legend()
        st.pyplot(fig)

else:
    st.title("Error 404: Page Not Found, please select a valid page.")