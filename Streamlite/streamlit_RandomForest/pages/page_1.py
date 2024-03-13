"""Page 1

"""
import streamlit as st
import sklearn as sk




def main():

    st.title("Random-Forest regressor")
    st.write("Let's do some regression!")

    st.sidebar.title("RM-Options")

    rm_option_list = [2, 8, 16]
    st.sidebar.radio("max_depth", rm_option_list, key="rm_key_options")
    st.sidebar.slider("random_state", 0, 200, key="slider")






if __name__ == "__main__":
     main()