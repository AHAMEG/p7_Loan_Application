# ------------------------------------
# import packages
# ------------------------------------
import pandas
from sklearn.manifold import trustworthiness
import random
from sklearn.manifold import TSNE
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
        content = json.loads(response.content.decode('utf-8'))  #
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
        targ_all_cust = (pd.Series(content['target_all_cust']).rename('TARGET'))
        target_select_cust = (pd.Series(content['target_selected_cust']).rename('TARGET'))
        target_neig = (pd.Series(content['target_neigh']).rename('TARGET'))

        data_all_customers = pd.DataFrame(content['data_all_cust'])
        data_selected_customer = pd.DataFrame(content['data_selected_cust'])
        data_neig = pd.DataFrame(content['data_neigh'])

        return targ_all_cust, target_select_cust, data_all_customers, data_selected_customer, data_neig, target_neig  # x_neig, y_neig, fea_sel, tar_sel, tar_cust

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
        tar_sel = json_normalize(content['target_sel'])
        dat_sel = pd.DataFrame(content['data_selected'])
        return dat_sel, feat_sel, tar_sel

    #############################################################################
    #                          Selected id
    #############################################################################
    # list of customer's ID's
    cust_id = get_id_list()
    # Selected customer's ID
    selected_id = st.sidebar.selectbox('Select customer ID from list:', cust_id, key=18)
    st.write('Your selected ID = ', selected_id)

    def get_list_display_features(f, def_n, key):
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

    # Affiche les valeurs des clients en fonctions de deux paramètres en montrant leur classe
    # Compare l'ensemble des clients par rapport aux plus proches voisins et au client choisi.
    # X = données pour le calcul de la projection
    # ser_clust = données pour la classification des points (2 classes) (pd.Series)
    # n_display = items à tracer parmi toutes les données
    # plot_highlight = liste des index des plus proches voisins
    # X_cust = pd.Series des data de l'applicant customer
    # figsize=(10, 6)
    # size=10
    # fontsize=12
    # columns=None : si None, alors projection sur toutes les variables, si plus de 2 projection

    # (data_all_cust, target_all_cust, 200, data_neigh, data_selected_cust,
    # figsize=(10, 6), size = 20, fontsize = 16, columns = data_all_cust.columns[5:7])

    def plot_scatter_projection(X, ser_clust, n_display, plot_highlight, X_cust,
                                figsize=(10, 6), size=10, fontsize=12, columns=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        X_all = X #pd.concat([X, X_cust], axis=0)
        ind_neigh = list(plot_highlight.index)
        customer_idx = X_cust.index[0]

        columns = X_all.columns if columns is None else columns

        st.write('X_all :', X_all.head(5))
        st.write('ind_neigh :', ind_neigh)
        st.write('customer_idx :', customer_idx)
        st.write('columns', columns)

        # if len(columns) == 2:
        #     # if only 2 columns passed
        df_data = X_all.loc[:, columns]
        st.write('df_data', df_data.head(5))
        ax.set_title('Two features compared', fontsize=fontsize + 2, fontweight='bold')
        ax.set_xlabel(columns[0], fontsize=fontsize)
        ax.set_ylabel(columns[1], fontsize=fontsize)
        #
        # elif len(columns) > 2:
        #     # if more than 2 columns passed
        #     # Compute T-SNE projection
        #     tsne = TSNE(n_components=2, random_state=14)
        #     df_proj = pd.DataFrame(tsne.fit_transform(X_all),
        #                            index=X_all.index,
        #                            columns=['t-SNE' + str(i) for i in range(2)])
        #     trustw = trustworthiness(X_all, df_proj, n_neighbors=5, metric='euclidean')
        #     trustw = "{:.2f}".format(trustw)
        #     ax.set_title(f't-SNE projection (trustworthiness={trustw})',
        #                  fontsize=fontsize + 2, fontweight='bold')
        #     df_data = df_proj
        #     ax.set_xlabel("projection axis 1", fontsize=fontsize)
        #     ax.set_ylabel("projection axis 2", fontsize=fontsize)
        #
        # else:
        #     # si une colonne seulement
        #     df_data = pd.concat([X_all.loc[:, columns], X_all.loc[:, columns]], axis=1)
        #     ax.set_title('One feature', fontsize=fontsize + 2, fontweight='bold')
        #     ax.set_xlabel(columns[0], fontsize=fontsize)
        #     ax.set_ylabel(columns[0], fontsize=fontsize)
        #
        # Showing points, cluster by cluster
        colors = ['green', 'red']
        for i, name_clust in enumerate(ser_clust.unique()):
            ind = ser_clust[ser_clust == name_clust].index

            if n_display is not None:
                display_samp = random.sample(set(list(X.index)), 200)
                ind = [i for i in ind if i in display_samp]
            # plot only a random selection of random sample points
            ax.scatter(df_data.loc[ind].iloc[:, 0],
                       df_data.loc[ind].iloc[:, 1],
                       s=size, alpha=0.7, c=colors[i], zorder=1,
                       label=f"Random sample ({name_clust})")
            # # plot nearest neighbors
            # ax.scatter(df_data.loc[ind_neigh].iloc[:, 0],
            #            df_data.loc[ind_neigh].iloc[:, 1],
            #            s=size * 5, alpha=0.7, c=colors[i], ec='k', zorder=3,
            #            label=f"Nearest neighbors ({name_clust})")

            # # plot the applicant customer
            # ax.scatter(df_data.loc[customer_idx].iloc[:, 0],  # :customer_idx
            #             df_data.loc[customer_idx].iloc[:, 1],  # :customer_idx
            #             s=size * 10, alpha=0.7, c='yellow', ec='k', zorder=10,
            #             label="Applicant customer")

            ax.tick_params(axis='both', which='major', labelsize=fontsize)

        ax.legend(prop={'size': fontsize - 2})

        return fig


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
        # # data loading
        # with open('df.csv', 'rb') as file:
        #     df = joblib.load(file)
        # st.write("df :", df.head(50))
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
            fig, ax = plt.subplots(figsize=(15, 4))
            with st.spinner('Boxplot creation in progress...please wait.....'):
                # df loading
                with open('df.csv', 'rb') as file:
                    df = joblib.load(file)
                # Get Shap values for customer
                shap_vals_original = shap_val()
                shap_vals = shap_vals_original[0]
                # Get features names
                features = feat()
                # Get selected columns
                disp_box_cols = get_list_display_features(features, 2, key=45)
                # -----------------------------------------------------------------------------------------------
                # Get tagets and data for : all customers + Applicant customer + 20 neighbors of selected customer
                # -----------------------------------------------------------------------------------------------
                target_all_cust, target_selected_cust, data_all_cust, data_selected_cust, data_neigh, target_neigh = \
                    get_data_neigh(selected_id)

                # Target impuatation (0 : 'repaid (....), 1 : not repaid (....)
                # -------------------------------------------------------------
                target_all_cust = target_all_cust.replace({0: 'repaid (all_cust)',
                                                           1: 'not repaid (all_cust)'})
                target_selected_cust = target_selected_cust.replace({0: 'repaid (selected_cust)',
                                                                     1: 'not repaid (selected_cust)'})

                target_neigh = target_neigh.replace({0: 'repaid (neighbors)',
                                                     1: 'not repaid (neighbors)'})

                # st.write("target_all_cust :", target_all_cust)
                # st.write("target_selected_cust :", target_selected_cust)
                st.write("data_all_cust :", data_all_cust.head(50))
                # st.write("data_selected_cust :", data_selected_cust)
                # st.write("target neigh :", target_neigh)
                # st.write("data neigh :", data_neigh)

                # ------------------
                # All customers data
                # ------------------
                df_all_cust = pd.concat([data_all_cust[disp_box_cols], target_all_cust], axis=1)
                df_melt_all_cust = df_all_cust.reset_index()
                df_melt_all_cust.columns = ['index'] + list(df_melt_all_cust.columns)[1:]
                df_melt_all_cust = df_melt_all_cust.melt(id_vars=['index', 'TARGET'],
                                                         value_vars=disp_box_cols,
                                                         var_name="variables",
                                                         value_name="values")
                # st.write("df_melt_all_cust modifiée:", df_melt_all_cust)
                # ----------------------
                #  All customers boxplot
                # ----------------------
                sns.boxplot(data=df_melt_all_cust, x='variables', y='values',
                            hue='TARGET', linewidth=1, width=0.4,
                            palette=['tab:green', 'tab:red'], showfliers=False,
                            saturation=0.5, ax=ax)

                # ------------------------------
                # Get 20 neighbors personal data
                # ------------------------------
                df_neigh = pd.concat([data_neigh[disp_box_cols], target_neigh], axis=1)
                # st.write("df_neigh :", df_neigh.head(10))
                df_melt_neigh = df_neigh.reset_index()
                df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
                df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],
                                                   value_vars=disp_box_cols,
                                                   var_name="variables",  # "variables",
                                                   value_name="values")
                # st.write("df_melt_neigh:", df_melt_neigh)

                # -------------------------------------------
                # 20 Applicant customer Neighbors swarmplot
                # -------------------------------------------
                sns.swarmplot(data=df_melt_neigh, x='variables', y='values',
                              hue='TARGET', linewidth=1,
                              palette=['darkgreen', 'darkred'], marker='o', edgecolor='k', ax=ax)

                # -----------------------
                # Applicant customer data
                # -----------------------
                df_selected_cust = pd.concat([data_selected_cust[disp_box_cols], target_selected_cust], axis=1)
                # st.write("df_sel_cust :", df_sel_cust)
                df_melt_sel_cust = df_selected_cust.reset_index()
                df_melt_sel_cust.columns = ['index'] + list(df_melt_sel_cust.columns)[1:]
                df_melt_sel_cust = df_melt_sel_cust.melt(id_vars=['index', 'TARGET'],
                                                         value_vars=disp_box_cols,
                                                         var_name="variables",
                                                         value_name="values")
                # st.write("df_melt_selected_cust:", df_melt_sel_cust)
                # -----------------------------
                #  Applicant customer swarmplot
                # -----------------------------
                sns.swarmplot(data=df_melt_sel_cust, x='variables', y='values',
                              linewidth=1, color='y', marker='o', size=10,
                              edgecolor='k', label='Applicant Customer', ax=ax)


                # legend
                h, _ = ax.get_legend_handles_labels()
                ax.legend(handles=h[:5])

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.write(fig)  # st.pyplot(fig) # the same

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.markdown('_Dispersion of the main features for random sample,\
                 20 nearest neighbors and applicant customer_')

                expander = st.expander("Concerning the dispersion graph...")

                expander.write("These boxplots show the dispersion of the preprocessed features values\
                 used by the model to make a prediction. The green boxplot are for the customers that repaid \
                their loan, and red boxplots are for the customers that didn't repay it.Over the boxplots are\
                 superimposed (markers) the values\
                 of the features for the 20 nearest neighbors of the applicant customer in the training set. The \
                 color of the markers indicate whether or not these neighbors repaid their loan. \
                 Values for the applicant customer are superimposed in yellow.")

    if st.sidebar.checkbox("Two features compare", key=50):
        target_all_cust, target_selected_cust, data_all_cust, data_selected_cust, data_neigh, target_neigh = \
            get_data_neigh(selected_id)

        # df loading
        with open('df.csv', 'rb') as file:
            df = joblib.load(file)

        targ_all_cust = df["TARGET"]
        dataa_all_cust = df.drop(["index", "TARGET", "SK_ID_CURR"], axis=1)
        # st.write('data_all_cust shape', data_all_cust.shape)
        # st.write('data_all_cust', data_all_cust.head(2))
        # st.write('target_all_cust shape', target_all_cust.shape)
        # st.write('target_all_cust', target_all_cust)
        # st.write('data_neigh shape', data_neigh.shape)
        # st.write('data_neigh', data_neigh)
        # st.write('data_selected_cust shape', data_selected_cust.shape)
        # st.write('data_selected_cust', data_selected_cust)
        plot_scatter_projection(X=dataa_all_cust,
                                ser_clust=targ_all_cust, #.replace({0: 'repaid', 1: 'not repaid'}),
                                n_display=200,
                                plot_highlight=data_neigh,
                                X_cust=data_selected_cust,
                                figsize=(10, 6),
                                size=20,
                                fontsize=16,
                                columns=dataa_all_cust.columns[5:7])

if __name__ == "__main__":
    main()
