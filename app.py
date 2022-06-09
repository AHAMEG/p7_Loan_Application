# Load librairies
import streamlit as st
from PIL import Image
import numpy as np
import requests
import os
import sys
import joblib
import imblearn
import matplotlib.pyplot as plt
import dill
import seaborn as sns
import pandas as pd
import sklearn as sk
from flask import Flask, request, jsonify, render_template
import json
import missingno as msno
from sklearn.neighbors import NearestNeighbors
import shap
import time
import pickle
# -------------------------------------------------------------------------------------------
#                                           loadings
# -------------------------------------------------------------------------------------------
# Model loading
# -------------
@st.cache
def model():
    path = os.path.join('scoring_credit_model.pkl')
    with open(path, 'rb') as file:
        best_model = joblib.load(file)
    return best_model


# find 20 nearest neighbors among the training set
def get_df_neigh(customer_id):
    # data loading
    with open('df.csv', 'rb') as file:
        df = joblib.load(file)

    # Target of All customers
    # -----------------------
    target_all = df["TARGET"]  # Matrice TARGET pour les 1000 clients

    # target of selected customer
    # ---------------------------
    customer_id = int(request.args.get('SK_ID_CURR'))
    df_cust = df[df['SK_ID_CURR'] == customer_id]
    target_cust = df_cust["TARGET"]

    # data of all customers
    # ---------------------
    # Get names of indexes for which column Stock has value "customer_id"
    indexnames = df[df['SK_ID_CURR'] == customer_id].index
    # Delete these row indexes from dataFrame
    df.drop(indexnames, inplace=True)
    data_all_cust = df
    target_all_cust = data_all_cust["TARGET"]
    data_all_cust = data_all_cust.drop(["index", "SK_ID_CURR", "TARGET"], axis=1)
    N=1000
    data_all_cust_rand = data_all_cust.sample(N)
    # data of selected customer
    # -------------------------
    data_cust = df_cust.drop(["index", "SK_ID_CURR", "TARGET"], axis=1)

    # target and data of neighbors
    # ---------------------------
    # fit nearest neighbors among the selection
    NN = NearestNeighbors(n_neighbors=20)
    NN.fit(data_all_cust_rand)  # data_all_cust
    idx = NN.kneighbors(X=data_cust,
                        n_neighbors=20,
                        return_distance=False).ravel()

    nearest_cust_idx = list(data_all_cust.iloc[idx].index)

    # data and target of neighbors
    # ----------------------------
    data_neigh = data_all_cust.loc[nearest_cust_idx, :]
    target_neigh = target_all_cust.loc[nearest_cust_idx]

    return target_all, target_cust, data_all_cust, data_cust, data_neigh, target_neigh  # features_neigh__, target_neigh__, features_cust, target_cust, features_all, target_all

# data loading
# -------------
@st.cache
def data():
    path_data_x = os.path.join('X.csv')
    path_data_x_id = os.path.join('X_id.csv')
    X = pd.read_csv(path_data_x)
    X_id = pd.read_csv(path_data_x_id)
    X = X.drop(['Unnamed: 0'], axis=1)
    X_id = X_id.drop(['Unnamed: 0'], axis=1)
    return X, X_id

# y_train loading
# ----------------------
@st.cache
def Y_train():
    path_data = os.path.join('y_train.csv')
    y_tr = pd.read_csv(path_data)
    y_tr = y_tr.drop(['Unnamed: 0'], axis=1)
    return y_tr

# x_train loading
# ----------------------
@st.cache
def x_train():
    path_data = os.path.join('X_train.csv')
    x_tr = pd.read_csv(path_data)
    x_tr = x_tr.drop(['Unnamed: 0'], axis=1)
    return x_tr

# threshold loading
# -------------
@st.cache
def threshold():
    # Threshold
    path = os.path.join('threshold_model.pkl')
    with open(path, 'rb') as file:
        tsholds = joblib.load(file)
    return tsholds

# shape_values
# -------------
@st.cache
def shap_val():
    path = os.path.join('shap_vals.pkl')
    with open(path, 'rb') as file:
        sh_val = joblib.load(file)
    return sh_val


# expected_values
# -------------
@st.cache
def expe_val():
    path = os.path.join('expected_values.pkl')
    with open(path, 'rb') as file:
        exp_val = joblib.load(file)
    return exp_val


# feature_names
# -------------
@st.cache
def feat_name():
    X, X_id = data()
    features = X.columns
    return features



###############################################################
# instantiate Flask object
app = Flask(__name__)


@app.route("/")
def index():
    return "API loaded, model and data loaded............"


# Customers id list
@app.route('/app/id/')
def ids_list():
    X, X_id = data()
    customers_id_list = pd.Series(list(X_id['SK_ID_CURR']))
    # Convert pd.Series to JSON
    customers_id_list_json = json.loads(customers_id_list.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': customers_id_list_json})


#################### l'envoi des shap_vals au serveur ralenti le fonctionnement de GET pour l'affichage graphique #####

# SHAP values
@app.route('/app/shap_val/')
def val_shap():
    shap_vals = pd.Series(list(shap_val()))
    # Convert pd.Series to JSON
    shap_val_json = json.loads(shap_vals.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': shap_val_json})


# features
@app.route('/app/feat/')
def feature_name():
    f = pd.Series(feat_name())
    # Convert pd.Series to JSON
    feat_json = json.loads(f.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': feat_json})


# expected values
@app.route('/app/exp_val/')
def val_expected():
    exp_vals = pd.Series(list(expe_val()))
    # Convert pd.Series to JSON
    exp_val_json = json.loads(exp_vals.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': exp_val_json})


  # Customers Index
  # test local : http://127.0.0.1:5000/app/cust_index/?SK_ID_CURR=100002
@app.route('/app/cust_index/')
def customer_index():
    X, X_id = data()
    # Parse http request to get arguments (sk_id_cust)
    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    X, X_id = data()
    x_cust = X_id[X_id['SK_ID_CURR'] == selected_id_customer]
    customer_ind = int(x_cust.index[0])
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'index': customer_ind}
                   )


# Scoring
# Test local : http://127.0.0.1:5000/app/score/?SK_ID_CURR=100002
@app.route('/app/score/')
def scoring():
    # Parse http request to get arguments (sk_id_cust)
    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    X, X_id = data()
    x_cust = X_id[X_id['SK_ID_CURR'] == selected_id_customer]
    customer_ind = int(x_cust.index[0])
    x_cust = X[X.index == customer_ind]
    #x_cust = X.loc[selected_id_customer:selected_id_customer]
    clf_step = model()
    # Get the data for the customer (pd.DataFrame)
    score = clf_step.predict_proba(x_cust)[0][1]
    # Convert pd.Series to JSON
    #score_json = json.loads(score.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': score,
                    'index': customer_ind})

# data loading
def dataf():
    with open('df.csv', 'rb') as file:
        df = joblib.load(file)
        df = df.drop(["index"], axis=1, inplace=True)
    return df


# test local : http://127.0.0.1:5000/app/data_cust/?SK_ID_CURR=100002
@app.route('/app/data_cust/')
def selected_cust_data():
    X, X_id = data()
    X = pd.concat([X, X_id], axis=1)

    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    x_cust = X[X['SK_ID_CURR'] == selected_id_customer] # x_cust = X_id[X_id['SK_ID_CURR'] == selected_id_customer]
    # customer_ind = int(x_cust.index[0])
    # x_cust = data_selected[data_selected.index == customer_ind] # x_cust = X[X.index == customer_ind]
    # Convert pd.Series to JSO
    data_x_json = json.loads(x_cust.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': data_x_json})


# Test local : http://127.0.0.1:5000/app/neigh_cust/?SK_ID_CURR=100002
@app.route('/app/neigh_cust/')
def neigh_cust():
    customer_id = int(request.args.get('SK_ID_CURR'))
    # Parse the http request to get arguments (selected_id), return the nearest neighbors
    target_all_cust, target_selected_cust, data_all_cust, data_selected_cust, data_neigh, target_neigh = \
        get_df_neigh(customer_id)  # x_neigh_df, y_neigh, features_cust, target_cust, features_sel, target_sel

    target_all_cust_json = json.loads(target_all_cust.to_json())
    target_selected_cust_json = json.loads(target_selected_cust.to_json())

    data_all_cust_json = json.loads(data_all_cust.to_json())
    data_selected_cust_json = json.loads(data_selected_cust.to_json())

    data_neigh_json = json.loads(data_neigh.to_json())
    target_neigh_json = json.loads(target_neigh.to_json())

    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
                    'target_all_cust': target_all_cust_json,
                    'target_selected_cust': target_selected_cust_json,
                    'target_neigh': target_neigh_json,
                    'data_all_cust': data_all_cust_json,
                    'data_selected_cust': data_selected_cust_json,
                    'data_neigh': data_neigh_json},
    )


# Threshold
@app.route('/app/threshold/')
def thresholds():
    thresh = pd.Series(threshold())
    # Convert pd.Series to JSON
    thresh_json = json.loads(thresh.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': thresh_json})


# model_and_params
@app.route('/app/explainer/')
def explainer_shap():
    explainer = pd.Series(shap.TreeExplainer(model()))
    # Convert pd.Series to JSON
    explainer_json = json.loads(explainer.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': explainer_json})


# Feature importances
@app.route('/app/feat_imp/')
def send_feat_imp():
    x_tr = x_train()
    clf_step = model()
    feat_imp = pd.Series(clf_step.feature_importances_,
                         index= x_tr.columns).sort_values(ascending=False)
    # Convert pd.Series to JSON
    feat_imp_json = json.loads(feat_imp.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
                    'data': feat_imp_json})


# x_train
@app.route('/app/x_train/')
def data_x_tr():
    # get all data from X_train data
    x_tr = x_train()
    # and convert the data to JSON
    x_tr_json = json.loads(x_tr.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
                    'X_train': x_tr_json})

# y_train
@app.route('/app/y_train/')
def data_y_tr():
    # get all data from y_train data
    y_tr = Y_train()
    # and convert the data to JSON
    y_tr_json = json.loads(y_tr.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
                    'y_train': y_tr_json})



# Test local : http://127.0.0.1:5000/app/data_selected/?SK_ID_CURR=100002
@app.route('/app/data_selected/')
def df_selected():
    # Parse the http request to get arguments (selected_id), return the nearest neighbors
    data_selected, features_sel, target_sel = get_df_selected()
    # Convert the customer's data to JSON
    data_selected_json = json.loads(data_selected.to_json())
    features_sel_json = json.loads(features_sel.to_json())
    target_sel_json = json.loads(target_sel.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
                    'features_sel': features_sel_json,
                    'target_sel': target_sel_json,
                    'data_selected': data_selected_json},
    )




   # test
if __name__ == "__main__":
    app.run()
