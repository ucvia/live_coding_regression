import streamlit as st
import numpy as np


@st.cache_data
def generate_data(n=20, m=5, sigma=0.3, density=0.2, seed=123456):
    "Generates data matrix X and observations Y."
    np.random.seed(seed)

    # Generación del modelo
    beta_star = sigma*np.random.randn(m)

    # Seleccionar los indices de sparcidad
    idxs = np.random.choice(range(m), int((1-density)*m), replace=False)

    # Setteando los valores exactamente a 0
    for idx in idxs:
        beta_star[idx] = 0

    # Creando la matriz del modelo N(0, sigma=1)
    X = np.random.randn(n, m)

    # Agregando ruido ~ N(0, sigma)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=n)

    return X, Y.reshape(n, 1), beta_star.reshape(m, 1)
