"""Page 1

"""
import streamlit as st
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
import pandas as pd




def main():

    st.title("Random-Forest regressor")
    st.write("Let's do some regression!")

    st.sidebar.title("RM-Options")

    rm_option_list = [2, 8, 16]
    st.sidebar.radio("max_depth", rm_option_list, key="rm_key_options")
    st.sidebar.slider("random_state", 0, 200, key="slider")

    st.sidebar.markdown("""---""")
    st.sidebar.subheader("Subheader")
    st.sidebar.write("Text in subheader")


    ##############################################################
    rm_regress = RandomForestRegressor()

    pd_dataframe = pd.read_csv("./datasets/train.csv")
    data_X = pd_dataframe.iloc[:, 2:11]
    data_X  = data_X.rename(columns={"on road now": "on_road_now", "top speed": "top_speed"})
    data_y = pd_dataframe["current price"]

    pd_dataframe = pd.read_csv("./datasets/train.csv")

    st.dataframe(data_X)
    st.dataframe(data_y)

    count_cols = data_X.shape[0]
    print("anzahl:",count_cols)
    ##### Textset
    test_x = data_X.head(200)
    test_y = data_y.head(200)

    rm_regress.fit(data_X, data_y)
    print(rm_regress)
    #print(rm_regress.predict([ [798186,	3,	78945,	1,	2,	14,	177, 73, 123] ]))
    print(f"score: {rm_regress.score(test_x, test_y)}")
    ##############################################################








if __name__ == "__main__":
     main()