import streamlit as st
import matplotlib.pyplot as plt

st.sidebar.title("Menu")
page = st.sidebar.selectbox('Choose Page', ["Home", "Sales", "Customers","Inventory","Reports"])

if page == "Home":
    st.title("Welcome to the Paintball Company Dashboard")
    st.write("This dashboard provides insights into various aspects of the business.")
    if st.button("Click to see more"):
        st.write("More features coming soon... Stay tuned!")
    st.image("Paintball2.jpg", caption="Paintball Game", use_container_width=True)

elif page == "Sales":
    st.title("Sales Overview")
    st.write("Here you can find information about sales performance.")
    if st.button("Show Sales Chart"):
        st.write("Sales Data will be updated soon.. Thank you for your patience!")

elif page == "Customers":
    st.title("Customer Insights")
    st.write("Analyze customer demographics and behavior.")
    if st.button("Load Customer Data"):
        st.write("Customer Data will be updated soon.. Thank you for your patience!")

elif page == "Inventory":   
    st.title("Inventory Management")
    st.write("Monitor stock levels and manage inventory effectively.")
    if st.button("Check Inventory Status"):
        st.write("Inventory Data will be updated soon.. Thank you for your patience!")

elif page == "Reports":
    st.title("Business Reports")
    st.write("This page will generate various business reports.")
    if st.button("Generate Sales Report"):
        st.write("Sales Report will be generated soon.. Thank you for your patience!")
    if st.button("Generate Customer Report"):
        st.write("Customer Report will be generated soon.. Thank you for your patience!")

else:
    st.title("Page not found - Please select a valid option from the menu. Code 404")