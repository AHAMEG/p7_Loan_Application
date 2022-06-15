# ------------------------------------
# import packages
# ------------------------------------
import pandas
import PIL.Image
import requests
import json
from pandas import json_normalize
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from flask import Flask, request, jsonify, render_template
from lightgbm import LGBMClassifier
import sklearn as sk
import joblib
import imblearn
import seaborn as sns
import missingno as msno
import sys
import os
import shap
import time
import pickle
from random import sample
from shap import TreeExplainer, Explanation
from shap.plots import waterfall
import matplotlib.pyplot as plt
from tkinter import *


# ----------------------------------------------------
# main function
# ----------------------------------------------------
def main():
    # ------------------------------------------------
    # local API (à remplacer par l'adresse de l'application déployée)
    # -----------------------------------------------
    API_URL = "http://127.0.0.1:5000/app/"
    # Local URL: http://localhost:8501
    # -----------------------------------------------
    # Configuration of the streamlit page
    # -----------------------------------------------
    st.set_page_config(page_title='Loan application scoring dashboard',
                       page_icon='🧊',
                       layout='centered',
                       initial_sidebar_state='auto')
    # Display the title
    st.title('Loan application scoring dashboard')
    st.subheader("ABDELKARIM HAMEG - Data Scientist")
    # Display the LOGO
    path = "lOGO.jpg"
    image = PIL.Image.open(path)
    st.sidebar.image(image, width=250)
    # Display the loan image
    path = "loan.jpg"
    image = PIL.Image.open(path)
    st.image(image, width=100)

    ###############################################################################
    #                      LIST OF API REQUEST FUNCTIONS
    ###############################################################################
    # Get list of ID (cached)
    @st.cache(suppress_st_warning=True)
    def get_id_list():
        # URL of the sk_id API
        id_api_url = API_URL + "id/"
        # Requesting the API and saving the response
        response = requests.get(id_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        id_customers = pd.Series(content['data']).values
        return id_customers

    # Get selected customer's data (cached)
    # local test api : http://127.0.0.1:5000/app/data_cust/?SK_ID_CURR=100002
    data_type = []

    @st.cache
    def get_selected_cust_data(selected_id):
        # URL of the sk_id API
        data_api_url = API_URL + "data_cust/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)  # decode('utf-8')
        df_cus = json_normalize(content['data'])  # Results contain the required data
        return df_cus

    # Get list of customers index
    # local test api : http://127.0.0.1:5000/app/cust_index/?SK_ID_CURR=100002
    @st.cache
    def customer_ind(selected_id):
        # URL of the sk_id API
        cust_index_api_url = API_URL + "cust_index/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(cust_index_api_url)
        # # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))  #
        # # Getting the values of "index" from the content
        ind = (content['index'])
        return ind

    # Get score (cached)
    @st.cache
    def get_score_model(selected_id):
        # URL of the sk_id API
        score_api_url = API_URL + "score/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(score_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # Getting the values of "ID" from the content
        score_model = (content['data'])
        return score_model

    # Get list of shap_values (cached)
    # local test api : http://127.0.0.1:5000/app/shap_val//?SK_ID_CURR=10002
    @st.cache
    def values_shap(selected_id):
        # URL of the sk_id API
        shap_values_api_url = API_URL + "shap_val/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(shap_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        shapvals = pd.Series(content['data']).values
        return shapvals

    # Get all data in train set (cached)
    @st.cache
    def get_data_x_tr():
        # URL of the scoring API
        data_tr_api_url = API_URL + "x_train/"
        # save the response of API request
        response = requests.get(data_tr_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        x_tra = (pd.DataFrame(content['X_train']))
        return x_tra

    # Get all data in train set (cached)
    @st.cache
    def get_data_y_tr():
        # URL of the scoring API
        data_tr_api_url = API_URL + "y_train/"
        # save the response of API request
        response = requests.get(data_tr_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        y_tra = (pd.Series(content['y_train']['TARGET']).rename('TARGET'))
        return y_tra

    #############################################
    #############################################
    # Get list of expected values (cached)
    @st.cache
    def values_expect():
        # URL of the sk_id API
        expected_values_api_url = API_URL + "exp_val/"
        # Requesting the API and saving the response
        response = requests.get(expected_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        expect_vals = pd.Series(content['data']).values
        return expect_vals

    # Get list of feature names
    @st.cache
    def feat():
        # URL of the sk_id API
        feat_api_url = API_URL + "feat/"
        # Requesting the API and saving the response
        response = requests.get(feat_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        features_name = pd.Series(content['data']).values
        return features_name

    # Get threshold (cached)
    @st.cache
    def get_thresh_model():
        # URL of the sk_id API
        threshold_api_url = API_URL + "threshold/"
        # Requesting the API and saving the response
        response = requests.get(threshold_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # Getting the values of "ID" from the content
        thresh_model = content['data']["0"]
        return thresh_model

    # Get explainer (cached)
    @st.cache
    def get_shap_explainer():
        # URL of the sk_id API
        explainer_shap_api_url = API_URL + "explainer/"
        # Requesting the API and saving the response
        response = requests.get(explainer_shap_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        explainer_shap_json = pd.Series(content['data']).values
        return explainer_shap_json

    # Get the list of feature importances (according to lgbm classification model)
    @st.cache
    def get_features_importances():
        # URL of the aggregations API
        feat_imp_api_url = API_URL + "feat_imp/"
        # Requesting the API and save the response
        response = requests.get(feat_imp_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    # Get data from 20 nearest neighbors in train set (cached)
    @st.cache
    def get_data_neigh(selected_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        neight_data_api_url = API_URL + "neigh_cust/?SK_ID_CURR=" + str(selected_id)
        # save the response of API request
        response = requests.get(neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        x_neig = pd.DataFrame(content['X_neigh'])
        y_neig = (pd.Series(content['y_neigh']).rename('TARGET'))
        fea_sel = pd.DataFrame(content['features_sel'])   # pd.Series(content['features_sel'])  # Results contain the required data
        tar_sel = (pd.Series(content['target_sel']).rename('TARGET'))
        tar_cust = (pd.Series(content['target_cust']).rename('TARGET'))
        return x_neig, y_neig, fea_sel, tar_sel, tar_cust

    # Get data from 20 nearest neighbors in train set (cached)
    @st.cache
    def get_data_select():
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        select_data_api_url = API_URL + "data_selected/"
        # save the response of API request
        response = requests.get(select_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        feat_sel = pd.DataFrame(content['features_sel'])
        targ_sel = json_normalize(content['target_sel'])
        dat_sel = pd.DataFrame(content['data_selected'])
        return dat_sel, feat_sel, targ_sel

    #############################################################################
    #                          Selected id
    #############################################################################
    # list of customer's ID's
    cust_id = get_id_list()
    # Selected customer's ID
    selected_id = st.sidebar.selectbox('Select customer ID from list:', cust_id, key=18)
    st.write('Your selected ID = ', selected_id)

    def get_list_display_features(shap_va, f, def_n, key):
        all_feat = f
        n = st.slider("Nb of features to display",
                      min_value=2, max_value=40,
                      value=def_n, step=None, format=None, key=key)

        disp_cols = list(get_features_importances().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect(
            'Choose the features to display:',
            sorted(all_feat),
            default=disp_cols, key=key)
        return box_cols


    # shape_values
    @st.cache
    def shap_val():
        path = os.path.join('shap_vals.pkl')
        with open(path, 'rb') as file:
            sh_val = joblib.load(file)
        return sh_val

    ############################################################################
    #                           Graphics Functions
    ############################################################################
    # Global SHAP SUMMARY
    @st.cache
    def shap_summary():
        return shap.summary_plot(shap_vals, feature_names=features)

    # Local SHAP Graphs
    @st.cache
    def waterfall_plot(nb, ft):
        return shap.plots._waterfall.waterfall_legacy(expected_val[1], shap_vals[index_cust],
                                                      max_display=nb, feature_names=ft)

    # Local SHAP Graphs
    @st.cache(allow_output_mutation=True)  #
    def force_plot(index_cust):
        shap.initjs()
        return shap.force_plot(expected_val[1], shap_vals[index_cust], matplotlib=True)

    # Local SHAP Graphs
    @st.cache
    def summary_plot():
        x_tr = get_data_x_tr()
        return shap.summary_plot(shap_vals, features=x_tr, feature_names=features)  # .sample(1000)

    # Gauge Chart
    @st.cache
    def gauge_plot(scor, th):
        scor = int(scor * 100)
        th = int(th * 100)

        if scor >= th:
            couleur_delta = 'red'
        elif scor < th:
            couleur_delta = 'Orange'

        if scor >= th:
            valeur_delta = "red"
        elif scor < th:
            valeur_delta = "green"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scor,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Selected Customer Score", 'font': {'size': 25}},
            delta={'reference': int(th), 'increasing': {'color': valeur_delta}},
            gauge={
                'axis': {'range': [None, int(100)], 'tickwidth': 1.5, 'tickcolor': "black"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, int(th)], 'color': 'lightgreen'},
                    {'range': [int(th), int(scor)], 'color': couleur_delta}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 1,
                    'value': int(th)}}))

        fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})
        return fig

    def boxplot():
        fig = sns.boxplot(x="variables", y="values", hue="TARGET",
                          data=df_melt_neigh, palette="Set2", fliersize=0)
        return fig

    ##############################################################################
    #                         Customer's data checkbox
    ##############################################################################
    if st.sidebar.checkbox("Customer's data"):
        st.markdown('data of the selected customer :')
        data_selected_cust = get_selected_cust_data(selected_id)
        data_selected_cust.columns = data_selected_cust.columns.str.split('.').str[0]
        st.write(data_selected_cust)
    ##############################################################################
    #                         Model's decision checkbox
    ##############################################################################
    if st.sidebar.checkbox("Model's decision", key=38):
        # Get score
        score = get_score_model(selected_id)
        # Threshold model
        threshold_model = get_thresh_model()
        # Display score (default probability)
        st.write('Default probability : {:.0f}%'.format(score * 100))
        # Display default threshold
        st.write('Default model threshold : {:.0f}%'.format(threshold_model * 100))  #
        # Compute decision according to the best threshold (False= loan accepted, True=loan refused)
        if score >= threshold_model:
            decision = "Loan rejected"
        else:
            decision = "Loan granted"
        st.write("Decision :", decision)
        ##########################################################################
        #              Display customer's gauge meter chart (checkbox)
        ##########################################################################
        figure = gauge_plot(score, threshold_model)
        st.write(figure)
        # Plot the graph on the dashboard
        # st.pyplot(plt.gcf())
        # Add markdown
        st.markdown('_Gauge meter plot for the applicant customer._')
        expander = st.expander("Concerning the classification model...")
        expander.write("The prediction was made using the Light Gradient Boosting classifier Model")
        expander.write("The default model is calculated to maximize air under ROC curve => maximize \
                                        True Positives rate (TP) detection and minimize False Negatives rate (FP)")

        ##########################################################################
        #                 Display local SHAP waterfall checkbox
        ##########################################################################
        if st.checkbox('Display local interpretation', key=25):
            with st.spinner('SHAP waterfall plots displaying in progress..... Please wait.......'):
                # Get expected values
                expected_val = values_expect()
                # #st.write('expected value =', expected_val[1])
                # Get Shap values for customer
                shap_vals_original = shap_val()
                shap_vals = shap_vals_original[0]
                # Get index customer
                index_cust = customer_ind(selected_id)
                # Get features names
                features = feat()
                # st.write(features)
                # st.write(features)
                nb_features = st.slider("Number of features to display",
                                        min_value=2,
                                        max_value=50,
                                        value=10,
                                        step=None,
                                        format=None,
                                        key=14)
                # draw the waterfall graph (only for the customer with scaling
                waterfall_plot(nb_features, features)
                plt.gcf()
                st.pyplot(plt.gcf())
                # Add markdown
                st.markdown('_SHAP waterfall Plot for the applicant customer._')
                #     # Add details title
                expander = st.expander("Concerning the SHAP waterfall  plot...")
                #     # Add explanations
                expander.write("The above waterfall  plot displays \
            #     explanations for the individual prediction of the applicant customer.\
                  The bottom of a waterfall plot starts as the expected value of the model output \
                  (i.e. the value obtained if no information (features) were provided), and then \
            #     each row shows how the positive (red) or negative (blue) contribution of \
            #     each feature moves the value from the expected model output over the \
            #     background dataset to the model output for this prediction.")

        ##########################################################################
        #              Display feature's distribution (Boxplots)
        ##########################################################################
        if st.checkbox('show features distribution by class', key=20):
            st.header('Boxplots of the main features')
            fig, ax = plt.subplots(figsize=(25, 5))
            with st.spinner('Boxplot creation in progress...please wait.....'):
                # df loading
                with open('df.csv', 'rb') as file:
                    df = joblib.load(file)
                # y_train loading
                y_tr = get_data_y_tr()
                y_tr = y_tr.replace({0: 'repaid (global)',
                                     1: 'not repaid (global)'})
                # Get Shap values for customer
                shap_vals_original = shap_val()
                shap_vals = shap_vals_original[0]
                # Get features names
                features = feat()
                # Get selected columns
                disp_box_cols = get_list_display_features(shap_vals, features, 2, key=45)
                # ---------------------------------------------
                # Get 20 neighbors personal data (preprocessed)
                # ---------------------------------------------
                x_neigh, y_neigh, x_sel, targ_sel, targ_cust = get_data_neigh(selected_id)  # , features_cust, target_cust
                y_neigh = y_neigh.replace({0: 'repaid (neighbors)',
                                           1: 'not repaid (neighbors)'})
                # st.write("y_neigh :", y_neigh)
                # st.write("x_neigh :", x_neigh[disp_box_cols])
                df_neigh = pd.concat([x_neigh[disp_box_cols], y_neigh], axis=1)
                # st.write("df_neigh :", df_neigh.head(10))
                df_melt_neigh = df_neigh.reset_index()
                df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
                df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],
                                                   value_vars=disp_box_cols,
                                                   var_name="variables",  # "variables",
                                                   value_name="values")
                # st.write("df_cust:", df_melt_neigh)

                sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET',
                              linewidth=1, size=10, palette=['darkred', 'darkgreen'], marker='o', edgecolor='k', ax=ax)
                # ---------------------------------------------
                # random sample of customers :
                # ---------------------------------------------
                targ_sel = targ_sel.replace({0: 'repaid (random_sample_customer)',
                                             1: 'not repaid (random_sample_customer)'})
                df_sel_cust = pd.concat([x_sel[disp_box_cols], targ_sel], axis=1)
                # st.write("df_sel_cust :", df_sel_cust)
                df_melt_sel_cust = df_sel_cust.reset_index()
                df_melt_sel_cust.columns = ['index'] + list(df_melt_sel_cust.columns)[1:]
                df_melt_sel_cust = df_melt_sel_cust.melt(id_vars=['index', 'TARGET'],
                                                         value_vars=disp_box_cols,
                                                         var_name="variables",
                                                         value_name="values")
                # st.write("df_sel_cust:", df_melt_sel_cust)
                sns.boxplot(data=df_melt_sel_cust, x='variables', y='values', hue='TARGET', linewidth=1,
                            width=0.4, palette=['tab:red', 'tab:green'], showfliers=False, saturation=0.5,
                            ax=ax)
                # ----------------------------------------------
                # Applicant customer
                # ----------------------------------------------
                data_applicant_cust = get_selected_cust_data(selected_id)
                data_applicant_cust.columns = data_applicant_cust.columns.str.split('.').str[0]
                data_applicant_cust = data_applicant_cust
                # st.write("x_cust type :", type(data_applicant_cust))
                # st.write("x_cust :", data_applicant_cust)

                targ_cust = targ_cust.replace({0: 'repaid (applicant customer)',
                                               1: 'not repaid (applicant customer)'})
                targ_cust = targ_cust.to_frame('TARGET')
                # st.write("y_cust type :", type(targ_cust))
                # st.write("y_cust :", targ_cust)

                data_melt_applicant_cust = data_applicant_cust.reset_index()
                targ_melt_cust = targ_cust.reset_index()
                # st.write("data_melt_applicant_cust :", data_melt_applicant_cust)
                # st.write("targ_melt_cust :", targ_melt_cust)

                data_applicant_cust = pd.concat([data_melt_applicant_cust[disp_box_cols], targ_melt_cust], axis=1)
                data_applicant_cust = data_applicant_cust.drop(["index"], axis=1, inplace=False)
                # st.write("df_applicant_cust :", data_applicant_cust)
                data_applicant_cust = data_applicant_cust.reset_index()
                data_applicant_cust.columns = ['index'] + list(data_applicant_cust.columns)[1:]
                data_applicant_cust = data_applicant_cust.melt(id_vars=['index', 'TARGET'],
                                                               value_vars=disp_box_cols,
                                                               var_name="variables",
                                                               value_name="values")
                # st.write("df_melt_sel_cust modifiée:", data_applicant_cust)

                sns.swarmplot(data=data_applicant_cust, x='variables', y='values', linewidth=1, color='y',
                              marker='o', size=15, edgecolor='k', label='applicant customer', ax=ax)

                # legend
                h, _ = ax.get_legend_handles_labels()
                ax.legend(handles=h[:5])

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.write(fig)  # st.pyplot(fig) # the same

                plt.xticks(rotation=20, ha='right')
                plt.show()


if __name__ == "__main__":
    main()
