import streamlit as st
import numpy as np


class Dataset:
    def __init__(self, seed=123456) -> None:
        self.seed = seed
        self._X = None
        self._y = None
        self._beta = None

    # @st.cache_data # decorators st.cache_data(staticmethod(generate_data))
    def generate_data(self, n=20, m=5, sigma=0.3, density=0.2, _return=True):
        "Generates data matrix X and observations Y."
        np.random.seed(self.seed)

        # Generación del modelo
        beta_star = sigma * np.random.randn(m)

        # Seleccionar los indices de sparcidad
        idxs = np.random.choice(range(m), int((1 - density) * m), replace=False)

        # Setteando los valores exactamente a 0
        for idx in idxs:
            beta_star[idx] = 0

        # Creando la matriz del modelo N(0, sigma=1)
        self._X = np.random.randn(n, m)

        # Agregando ruido ~ N(0, sigma)
        self._y = self._X.dot(beta_star) + np.random.normal(0, sigma, size=n)
        self._y = self._y.reshape(n, 1)

        self._beta = beta_star.reshape(m, 1)

        if _return:
            return self._X, self._y, self._beta
