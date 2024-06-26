import streamlit as st
import pandas as pd
import numpy as np

from lib.dataset import Dataset
from lib.models import OLS, Lasso
from lib.utils import plot_coefficients, plot_regularization_path


st.header("UCV demo Ridge/Lasso")

# Ingresando valor de n
with st.sidebar:
    st.subheader("Datos")
    _n = st.number_input("Inserte el número de instancias", min_value=1, max_value=300, value=100)
    # st.write("Número de instancias", _n)

    _m = st.number_input(
        "Inserte el número de características", min_value=5, max_value=300, value=20
    )
    # st.write("Número de instancias", _m)

    _sigma = st.number_input("Inserte el sigma de ruido", min_value=0.0, max_value=5.0, value=2.0)
    # st.write("Número de instancias", _sigma)

    _density = st.number_input(
        "Inserte la densidad", min_value=0.0, max_value=1.0, step=0.01, value=0.3
    )
    # st.write("Número de instancias", _density)

    _seed = st.number_input("Inserte la semilla", min_value=123456, max_value=2000000)
    # st.write("Número de instancias", _seed)

    st.subheader("Regularización")
    _lambda = st.number_input("Lambda=", min_value=0.0, max_value=1000.0, step=0.01)
    _step = st.number_input("step=", min_value=1.0, max_value=10.0, step=1.0)

    _lambda_path = st.select_slider("Lambda_path=", options=np.arange(0, 1000, _step), value=[min(np.arange(0, 1000, _step)), max(np.arange(0, 1000, _step))])

    _s = st.number_input("s=", min_value=0.0, max_value=100.0, step=0.01)


t1, t2, t3 = st.tabs(["Datos", "Experimento simple", "Camino de regularización"])

D = Dataset(seed=123456)

_X, _y, _beta = D.generate_data(n=_n, m=_m, sigma=_sigma, density=_density)

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
    _beta_lasso_s = modelo3.resolver(_s=_s)

    _df = pd.DataFrame.from_dict(
        {
            "w": _beta.reshape(_m),
            "OLS": _beta_ols.reshape(_m),
            "Constrained": _beta_lasso_s.reshape(_m),
            "Lagrangian": _beta_lasso_lambda.reshape(_m),
        }
    )

    st.text("Modelos optimales")

    st.dataframe(_df)

    st.text("Graficando los coeficientes")
    figure = plot_coefficients(df=_df)
    st.pyplot(fig=figure)


with t3:
    st.text("Gráficos")
    my_bar = st.progress(0, text="Calculando camino de regularización")

    lambdas = np.arange(_lambda_path[0], _lambda_path[1], _step)

    
    # st.text("\n".join(map(lambda x: str(x), lambdas)))
    # plot_regularization_path(lambda_values, w_values, df[np.abs(df.w)>0].index.values)

    modelo2 = Lasso(X=_X, y=_y)
    w_values = modelo2.path(values=lambdas, progress=my_bar, _lambda = True)

    # st.text(w_values)

    _df = pd.DataFrame.from_dict(
        {
            "w": _beta.reshape(_m),
        }
    )

    figure = plot_regularization_path(
        lambdas, w_values, _df[np.abs(_df.w) > 0].index.values
    )

    st.pyplot(fig=figure)
