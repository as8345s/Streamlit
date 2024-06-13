"""
Autor:           Alex.S
Description:     ---
Written in:      IDE PyCharm.
"""

import streamlit as st
import PIL
import time


def page_1() -> None:
    st.title("Data Exploration")
    st.write("This is the data exploration page.")


def page_2() -> None:
    st.title("Model Training")
    st.write("This is the model training page.")


def page_home() -> None:
    """ Home page.
    The first page you see when you start the application.
    :return: None
    """
    st.title("Home Page")
    st.write("Welcome to the home page!")


def main() -> None:
    """
    Start point of main programm.
    :return: None
    """

    # Directory for available pages.
    pages = {
        "Home": page_home,
        "First page": page_1,
        "Second page": page_2
    }

    # Trying out some of the Streamlit Widgets.
    # Streamlit offers many types of Widgets like buttons and text boxes for user input.
    # - Each widget has its own parameters

    def print_stuff():
        st.write('Callable')

    # Button:
    if st.button(label="Button", key="unique_key", help="help info", on_click=print_stuff):
        st.write('Button pressed')
    else:
        st.write('Button not pressed')

    checkbox_val = False
    if st.checkbox('Show and '):
        checkbox_val=True
    else:
        checkbox_val = False

    st.write(f'Check box value: {checkbox_val}')





if __name__ == "__main__":
    main()
