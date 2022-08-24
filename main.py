import streamlit as st
from streamlit_shap import st_shap
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import shap
from scipy.stats import t, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels
import matplotlib.pyplot as plt

from PIL import Image
import warnings

# warnings.filterwarnings("ignore")

####### Load Dataset #####################


URL = Path("D:/Users/laien/Documents/openClassRoom/formation DA/p10")


# URL = '.'

@st.cache
def load_data():
    # filenames = [filename for filename in filenames]
    data = pd.read_csv(URL / "data/billets.csv", sep=";")
    test_data = pd.read_csv(URL / "data/billets_production.csv")
    return data, test_data


@st.cache
def get_model():
    clf_path = URL / "models/lr_model.pickle"
    kmeans_path = URL / "models/kmeans_model.pickle"
    with open(clf_path, "rb") as f, open(kmeans_path, "rb") as f2:
        clf = pickle.load(f)
        kmean = pickle.load(f2)
    return clf, kmean


@st.cache
def get_logo():
    logo_path = URL / 'logo.svg'
    image = Image.open(logo_path)
    return image


@st.cache
def ols_summary_to_dataframe(results):
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals": pvals,
                               "coeff": coeff,
                               "conf_lower": conf_lower,
                               "conf_higher": conf_higher
                               })

    # Reordering...
    results_df = results_df[["coeff", "pvals", "conf_lower", "conf_higher"]]
    return results_df


# def st_shap(plot, height=None):
#     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#     components.html(shap_html, height=height)

# @st.cache
# def my_join(liste):
#     tmp=liste[0]
#     for elem in liste[1:]:
#         if elem != "intercept":
#             tmp+="+"+elem
#         else:
#             tmp+= "-1"
#     return(tmp)

# Configuration of the streamlit page
st.set_page_config(page_title='Dashboard DAN projet 10',
                   page_icon='random',
                   layout='centered',
                   initial_sidebar_state='auto')

# Display the title
st.title('Detection des faux billets')
st.subheader("Laien WU - Data Analyse project 10")
st.write("##")
st.write("##")
st.write("##")
logo = get_logo()
st.sidebar.image(logo, width=180)

Liste_partie = ["Data overview", "Regression linéaire", "Models explained", "Test model"]
side_bar = st.sidebar.selectbox("Select your topic: ", Liste_partie, index=0)

df, test_df = load_data()
clf, kmean = get_model()
df2 = df.dropna()

# Partie une (Overview)
if side_bar == Liste_partie[0]:
    list_select = ("True", "False", 'Both')
    select = st.sidebar.selectbox('Selection la nature des billets', list_select, index=2)
    dict_select = {"True": False, "False": True, "Both": pd.NA}
    data = df[df['is_genuine'] != dict_select[select]]
    data['is_genuine'] = data['is_genuine'].astype(int)

    st.markdown("### Description du jeu de donnée")
    st.dataframe(df.head())
    st.write(
        f"La table contient {df.shape[0]}, éléments, dont le nombre de billets authentiques s'élève à: **{df['is_genuine'].value_counts()[1]}** pièces "
        f"\n\n le nombre de billets de contrefaçon s'élève à: **{df['is_genuine'].value_counts()[0]}** pièces. ")

    st.write("##")
    st.write("##")
    # st.markdown("### Distribution des billets selon la nature des billets")
    fig = px.pie(data, "is_genuine", title="Distribution des billets selon la nature des billets")
    fig.update_layout(
        # margin=dict(l=20, r=20, t=20, b=20),
        autosize=False,
        showlegend=False
    )
    fig.update_traces(textinfo='label+value+percent')
    st.plotly_chart(fig)

    st.write("##")
    st.write("##")
    genuine = data[data['is_genuine'] == 1]
    counterfeit = data[data['is_genuine'] == 0]
    fig = make_subplots(rows=data.shape[1], cols=1)
    for i in range(len(genuine.columns[1:-1])):
        fig.add_trace(
            go.Box(x=genuine.iloc[:, i + 1],
                   # color=px.colors.DEFAULT_PLOTLY_COLORS[0]
                   marker_color="#636EFA",
                   name="Authentique"
                   # autocolorscale=False
                   ),
            row=i + 1, col=1)
        fig.add_trace(
            go.Box(x=counterfeit.iloc[:, i + 1],
                   marker_color='#EF553B',
                   name="Contrefaçon"),
            row=i + 1, col=1)
    fig.add_annotation(text=f'{genuine.columns[i + 1]}', row=i + 1, col=1, showarrow=False,
                       xref="x domain", yref="y domain", x=0.5, y=1.1)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=20))
    fig.update_layout(showlegend=False, title_text="Distribution des dimensions selon la nature du billet", height=1000)
    st.plotly_chart(fig, height=1000)

    st.dataframe(genuine)
    st.dataframe(counterfeit)

    st.dataframe(
        data.corr().dropna(axis=0, how="all").dropna(axis=1, how='all').style.background_gradient('coolwarm', 0.5))

    fig = sns.pairplot(data, hue='is_genuine', palette=['#EF553B', '#636EFA'])
    st.pyplot(fig)

# Partie deux (Régression linéaire)
if side_bar == Liste_partie[1]:
    list_select = ("True", "False", 'Both')
    select = st.sidebar.selectbox('Selection la nature des billets', list_select, index=2)
    dict_select = {"True": False, "False": True, "Both": pd.NA}
    data = df[df['is_genuine'] != dict_select[select]]
    data['is_genuine'] = data['is_genuine'].astype(int)

    liste_critere = st.multiselect("Selection des variables significatives",
                                   ["is_genuine", "diagonal", "height_left", "height_right", "margin_up", "length",
                                    "intercept"],
                                   ["is_genuine", "diagonal", "height_left", "height_right", "margin_up", "length",
                                    "intercept"])
    if 'intercept' in liste_critere:
        liste_critere2 = liste_critere.copy()
        liste_critere2.remove('intercept')
    else:
        liste_critere2 = liste_critere

    tmp = "+".join(liste_critere2)
    if "intercept" not in liste_critere:
        tmp += "-1"

    reg = smf.ols(formula=f'margin_low ~ {tmp}',
                  data=data).fit()
    st.markdown(f"le score $R^{2}$ du model vaut {reg.rsquared:.3f}, son $R^{2}$ ajusté vaut{reg.rsquared_adj:.3f}")
    st.dataframe(ols_summary_to_dataframe(reg).style.background_gradient(subset='pvals', vmax=1, vmin=0))

# Partie trois (Modélisation)
if side_bar == Liste_partie[2]:
    st.markdown(
        "#### Pour la suite du projet nous mettrons l'accent sur l'intégrité des données, c'est à dire que nous reprenons les données initiales et drop les nan au lieu de les imputer")
    st.write(df2.shape)
    # data = df[df['is_genuine'] != dict_select[select]]
    # data['is_genuine'] = data['is_genuine'].astype(int)

# Partie quatre ( test billets)
if side_bar == Liste_partie[3]:
    result_df = test_df.copy()
    result_df["res_regression"] = clf.predict(test_df.iloc[:, :-1]).astype(int)
    result_df["res_regression_prob"] = [elem[1] for elem in clf.predict_proba(test_df.iloc[:, :-1])]
    result_df["res_kmeans"] = kmean.predict(test_df.iloc[:, :-1]).astype(int)
    st.dataframe(result_df)
    X_train = df2.drop('is_genuine', axis=1).values
    y_train = df2['is_genuine'].values
    X_test = test_df.drop(['id'], axis=1).values
    features_names = df2.columns.drop('is_genuine')

    if st.checkbox("Détails d'analyse", key="22"):
        with st.spinner("Analyse détaillée en progression..."):
            liste_billet = test_df.index
            sample_idx = st.selectbox("veuillez choisir un des billets pour une analyse", liste_billet)
            lin_reg_explainer = shap.LinearExplainer(clf, X_train, feature_perturbation="correlation_dependent")
            shap_vals = lin_reg_explainer.shap_values(X_test[sample_idx].reshape(1, -1))[0]

            shap_values = lin_reg_explainer(X_test[sample_idx].reshape(1, -1))
            shap_values.feature_names = features_names.tolist()
            st_shap(shap.plots.waterfall(shap_values[0], max_display=len(features_names)))
            st.markdown('##')
            st.markdown('##')
            st_shap(shap.force_plot(lin_reg_explainer.expected_value,
                                    lin_reg_explainer.shap_values(X_test[sample_idx].reshape(1, -1))[0],
                                    feature_names=features_names,
                                    out_names="authenticité"))
            st.markdown('##')
            st.markdown('##')
            st_shap(shap.decision_plot(lin_reg_explainer.expected_value,
                                       lin_reg_explainer.shap_values(X_test[sample_idx].reshape(1, -1))[0],
                                       feature_names=features_names.tolist(),
                                       ))
