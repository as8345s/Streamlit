"""
Autor:           Alexander Schechtel
Description:     Basic usage of Streamlit. https://streamlit.io
Written in:      IDE PyCharm.

About this Code:
The app will show basic usage of Streamlit
"""

import streamlit as st
import PIL
import time
import sklearn as sk


def main() -> None:
    st.title("Welcome user")
    st.write("What is your name?")
    user_input = st.text_input("My name is... ")

    # Button
    button_clicked = st.button("Click Me!")
    # Handle button click
    if button_clicked:
        # Button click
        st.write("Button clicked!")
        if user_input:
            st.write("You entered:", user_input)
        else:
            st.write("You didn't enter any text.")

    st.write("# 😭")

    op = [1, 2]
    st.checkbox("Agree")


if __name__ == "__main__":
    main()
