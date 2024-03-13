"""
Autor:           Alexander Schechtel
Description:     Basic usage of Streamlit. https://streamlit.io
Written in:      IDE PyCharm.

About this Code:
The app will use RandomForest to solve a problem.

Streamlit will navigate automatically.
For that the project must have the structure:
   project/
        main.py
        pages/
            page_0.py
            page_n-1.py

"""

import PIL
import time
import sklearn as sk

# Contents of ~/my_app/main_page.py
import streamlit as st


def main():

    st.title("Welcome user")
    st.write("Let's do some regression!")
    st.sidebar.title("Sidebar titile")
    st.markdown("# Main page 🎈")
    #st.sidebar.markdown()
    st.sidebar.write("Sidebar text")




if __name__ == "__main__":
     main()