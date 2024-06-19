import streamlit as st
import pandas as pd
import numpy as np

from lib.dataset import generate_data
from lib.models import OLS, Lasso
from lib.utils import plot_coefficients


st.header('UCV demo Ridge/Lasso')

# Ingresando valor de n
with st.sidebar:
    st.subheader("Datos")
    _n = st.number_input("Inserte el número de instancias", min_value=1, max_value=300)
    # st.write("Número de instancias", _n)

    _m = st.number_input("Inserte el número de características", min_value=5, max_value=300)
    # st.write("Número de instancias", _m)

    _sigma = st.number_input("Inserte el sigma de ruido", min_value=0.0, max_value=5.0)
    # st.write("Número de instancias", _sigma)

    _density = st.number_input("Inserte la densidad", min_value=0.0, max_value=1.0, step=0.01)
    # st.write("Número de instancias", _density)

    _seed = st.number_input("Inserte la semilla", min_value=123456, max_value=2000000)
    # st.write("Número de instancias", _seed)

    st.subheader("Regularización")
    _lambda = st.number_input("Lambda=", min_value=0.0, max_value=100.0, step=0.01)
    _s = st.number_input("s=", min_value=0.0, max_value=100.0, step=0.01)
    

t1, t2, t3 = st.tabs(["Datos", "Experimento simple", "Camino de regularización"])

_X, _y, _beta  = generate_data(n=_n, m=_m, sigma=_sigma, density=_density, seed=_seed)

with t1:
    c1, c2, c3 = st.columns([0.6, 0.2, 0.2])

    with c1:
        st.text("X")
        st.dataframe(_X)

    with c2:
        st.text("y")
        st.dataframe(_y)

    with c3:
        st.text("Beta")
        st.dataframe(_beta)

with t2:
    modelo1 = OLS(X=_X, y=_y)
    _beta_ols = modelo1.resolver()

    modelo2 = Lasso(X=_X, y=_y)
    _beta_lasso_lambda = modelo2.resolver(_lambda=_lambda)

    modelo3 = Lasso(X=_X, y=_y)
    _beta_lasso_s = modelo2.resolver(_s=_s)

    _df = pd.DataFrame.from_dict(
        {
            "w": _beta.reshape(_m),
            "OLS": _beta_ols.reshape(_m),
            "Constrained": _beta_lasso_s.reshape(_m),
            "Lagrangian": _beta_lasso_lambda.reshape(_m)
        }
    )

    st.text("Modelos optimales")

    st.dataframe(_df)

    st.text("Graficando los coeficientes")
    figure = plot_coefficients(df=_df)
    st.pyplot(fig=figure)


with t3:
    st.text("Graficos")


