import streamlit as st
import matplotlib.pyplot as plt
#st.header("22009429")


#st.title("Welcome to Lab 1")
#st.header("Getting Started with Streamlit")
#st.subheader("Your First Streamlit App")
#st.text("This is a simple Streamlit application.")

#to on HTML mode
#st.markdown("""
# Markdown Example
## This is a second-level heading
### This is a third-level heading
#### This is a fourth-level heading
##### This is a fifth-level heading
###### This is a sixth-level heading
#<h1><b>This is an HTML heading</b></h1>

#AddEmoji
#            <b>:moon:</b>
#                :smile:
#            **bold**
#""", True)

#st.write(st)
#st.header("Streamlit Tutorial")
#st.video("https://www.youtube.com/watch?v=al776E73LjE&pp=ygUPcGFpbnRiYWxsIHNuYWtl")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox('Choose Page', ["Introduction", "Visual Picture", "Visual Graphs","Interactive Graph"])

if page == "Introduction":
    st.title("This is data visualization module")
    st.write("Streamlit is an open-source app framework for Machine Learning and Data Science projects.")

elif page == "Visual Picture":
    st.title("This is Visual Picture module")
    st.markdown(""" # I Hope your understand the difference""")

elif page == "Visual Graphs":
    st.title("This is Visual Graphs module")
    st.markdown(""" # I Hope your understand the difference""")
    st.image("Paintball G.png", caption="Paintball Graph", use_container_width=True)

elif page == "Interactive Graph":
    st.title("This is Interactive Graph")
    if st.button("Show Bar Chart"):
        data={'Apples': 10, 'Orange': 30, 'Nanas': 50}
        fig, ax = plt.subplots()
        ax.bar(data.keys(), data.values())
        st.pyplot(fig)

else:
    st.title("Do not left behind")

if page == 'Visual Picture':
    st.title("Visual Picture")
    if st.button("Display Image"):
        st.image("download.jpg", caption="Paintball Snake", use_container_width=True)