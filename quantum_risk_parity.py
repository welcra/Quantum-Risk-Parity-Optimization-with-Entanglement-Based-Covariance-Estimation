import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA

tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2022-01-01", end="2023-01-01")["Close"]
returns = data.pct_change().dropna().values

returns_norm = (returns - returns.mean(axis=0)) / returns.std(axis=0)

feature_map = ZZFeatureMap(feature_dimension=3, reps=2, entanglement="full")
backend = AerSimulator()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

kernel_matrix = quantum_kernel.evaluate(x_vec=returns_norm)
pca = PCA(n_components=len(tickers))
kernel_pca = pca.fit_transform(kernel_matrix)

pseudo_cov = np.cov(kernel_pca.T)

def portfolio_variance(w, cov):
    return w.T @ cov @ w

def risk_contributions(w, cov):
    port_var = portfolio_variance(w, cov)
    mrc = cov @ w
    return w * mrc / np.sqrt(port_var)

def risk_parity_loss(w, cov):
    rc = risk_contributions(w, cov)
    return np.sum((rc - rc.mean()) ** 2)

initial_w = np.ones(len(tickers)) / len(tickers)
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * len(tickers)

result = minimize(risk_parity_loss, initial_w, args=(pseudo_cov,),
                  method="SLSQP", bounds=bounds, constraints=constraints)
opt_weights = result.x
risk_contribs = risk_contributions(opt_weights, pseudo_cov)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(tickers, opt_weights)
plt.title("Quantum Risk Parity Weights")
plt.ylabel("Weight")

plt.subplot(1, 2, 2)
plt.bar(tickers, risk_contribs)
plt.title("Risk Contributions")
plt.ylabel("Contribution to Risk")

plt.tight_layout()
plt.show()

print("Optimal Weights (Quantum Risk Parity):", opt_weights)