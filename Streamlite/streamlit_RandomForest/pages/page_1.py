""" Page_1
Autor:           Alexander Schechtel
Description:     Basic usage of Streamlit. https://streamlit.io
Written in:      IDE PyCharm.

Dataset information:
Dataset 1: (regression dataset)
  SECOND HAND CARS DATA SET | REGRESSION
  Source: https://www.kaggle.com/datasets/mayankpatel14/second-hand-used-cars-data-set-linear-regression/data [14.03.2024]
  By MAYANK PATEL on Kaggle

Dataset 2:
"""

import streamlit as st
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import time


def view_dataset():

    pd_dataframe = pd.read_csv("./datasets/second_hand_cars_data.csv")

    st.dataframe(pd_dataframe)


def make_dataset():

    pd_dataframe = pd.read_csv("./datasets/second_hand_cars_data.csv")
    st.session_state['dataset_size'] = len(pd_dataframe.index)
    data_X = pd_dataframe.iloc[:, 2:11]
    data_X = data_X.rename(columns={"on road now": "on_road_now", "top speed": "top_speed"})
    data_y = pd_dataframe["current price"]

    if st.session_state['sort_out_val']:

        data_x_val = data_X.head( int(len(pd_dataframe.index) * (st.session_state['percent_val']/100)))
        data_y_val = data_y.head( int(len(pd_dataframe.index) * (st.session_state['percent_val']/100)))

        data_x_train = data_X.tail(int(len(pd_dataframe.index) * (1.0 - (st.session_state['percent_val'] / 100))))
        data_y_train = data_y.tail(int(len(pd_dataframe.index) * (1.0 - (st.session_state['percent_val'] / 100))))

        return data_x_val, data_y_val, data_x_train, data_y_train

    else:

        data_x_val = data_X.head(int(len(pd_dataframe.index) * (st.session_state['percent_val'] / 100)))
        data_y_val = data_y.head(int(len(pd_dataframe.index) * (st.session_state['percent_val'] / 100)))

        return data_x_val, data_y_val, data_X, data_y



def main():
    """Enter Page: page_1

    """

    # Text and description
    st.title("Random-Forest regressor")
    st.write("Let's do some regression!")
    st.write("About Random-Forest [...]")
    st.write(f"On the left side you can change the setting of the Random Forest model. \
             [...]")


    # Set RM options
    st.sidebar.title("RM-Options")
    rm_max_depth_option_list = [2, 8, 16]
    st.sidebar.radio("max_depth", rm_max_depth_option_list, key="rm_key_options")
    st.sidebar.slider("random_state", 0, 200, key="slider")

    string_code:str = (f"# Random-Forest Model \n"
                       f"model = RandomForestRegressor(max_depth={st.session_state['rm_key_options']}, random_state={st.session_state['slider']})")
    st.code(string_code, language='python')


    st.write(f"season content: {st.session_state}")

    st.sidebar.markdown("""---""")
    st.sidebar.subheader("Dataset")
    st.sidebar.write("Choose a dataset for training")

    radio_dataset_choice = ["SECOND HAND CARS DATA SET", "Dataset 2"]
    st.sidebar.radio("Dataset", radio_dataset_choice, key="dataset_choice", index=None)

    st.sidebar.number_input("validation data in (15-40%)", key="percent_val", min_value=15, max_value=40, value=20)
    st.sidebar.checkbox("Sort out validation data from dataset", key="sort_out_val")

    # This part of code will be organized later
    ##############################################################

    st.write("About the datasets...")
    st.write("Choose a dataset what will will be used for training.\
                The dataset will be displayed in a dataframe. You can choose how much of the Data will be used for training")

    # View the data
    radio_choice = st.session_state['dataset_choice']
    if radio_choice is None:
        st.write("Please select the dataset on the left sidebar")
    elif radio_choice == "SECOND HAND CARS DATA SET":
        st.write(f"You have selected the dataset: '{radio_choice}'")
        view_dataset()
    else:
        st.write("Please select the dataset on the left sidebar")

    st.write("When you have your dataset you can create it by clicking on the button")

    # Create the dataset after viewing it
    btn_make_dataset = st.button("Create dataset")
    st.code(f"# Create a dataset \n"
            f"X,y, X_val, y_val = make_dataset({radio_choice})\n")


    if btn_make_dataset:

        if radio_choice in radio_dataset_choice:
            val_x, val_y, train_x, train_y = make_dataset()
            st.code(f"# Dataset size: {st.session_state['dataset_size']} \n"
                    f"Len X: {len(train_x.index)} ({ (100 - st.session_state['percent_val']) 
                                          if st.session_state['sort_out_val'] else 100 } %), Len X_val: "
                    f"{len(val_x.index)} ({st.session_state['percent_val']} %)")

            st.write("Dataset created!")
        else:
            st.write("<Choose a dataset before you create one!>")

    """
    #print(rm_regress.predict([ [798186,	3,	78945,	1,	2,	14,	177, 73, 123] ]))
    print(f"score: {rm_regress.score(test_x, test_y)}")
    """

    training = st.button("Start Training", key="btn_start_training")
    if training:
        start_training(train_x, train_y)


def start_training(X,y):

    rm_regress = RandomForestRegressor(max_depth=st.session_state['rm_key_options'], random_state=st.session_state['slider'] )
    time_start = time.time()
    rm_regress.fit(X, y)
    time.sleep(3)
    st.write(f"Time elapsed: {time.time() - time_start}")


def validate():
    pass


if __name__ == "__main__":
    main()