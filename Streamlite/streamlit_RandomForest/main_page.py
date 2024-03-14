""" main_page.py
Autor:           Alexander Schechtel
Description:     Basic usage of Streamlit. https://streamlit.io
Written in:      IDE PyCharm.

About this Code:
The app will use RandomForest to solve a problem
and show the use of Streamlit.

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
    """Enter first page and execute code
    [More is coming]
    """

    st.title("Welcome user")
    st.write("Let's do some regression!")

    st.sidebar.title("Sidebar titile")
    #st.markdown()
    #st.sidebar.markdown()
    st.sidebar.write("Sidebar text")


if __name__ == "__main__":
     main()