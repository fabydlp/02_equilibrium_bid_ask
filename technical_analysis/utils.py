import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from typing import Tuple


class CopelandGalaiCalc:
    """
    Cálculo del spread de equilibrio y precios bid/ask bajo el modelo
    de Copeland & Galai (1983).

    El market maker fija un spread S tal que su beneficio esperado es cero
    frente a la posibilidad de operar con traders informados (probabilidad π).

    Soporta distribución Normal (μ, σ²) y Exponencial (λ).
    """

    # ---------- Distribución Normal ----------
    @staticmethod
    def bid_ask_normal(
        mu: float = 100,
        sigma: float = 10,
        pi: float = 0.3,
        initial_guess: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Equilibrium bid/ask for a normally-distributed fundamental value.

        Parameters
        ----------
        mu : float
            Mean of the fundamental value V.
        sigma : float
            Standard deviation of V.
        pi : float
            Probability that the counterparty is informed.
        initial_guess : float
            Initial guess for the non-linear solver.

        Returns
        -------
        bid : float
        ask : float
        spread : float  (ask − bid)
        """

        def _expected_profit(S: float) -> float:
            a = mu + S / 2          # ask quote
            b = mu - S / 2          # bid quote

            z_a = (a - mu) / sigma
            z_b = (mu - b) / sigma

            # Conditional expectations E[V | V > a] and E[V | V < b]
            E_V_gt_a = mu + sigma * norm.pdf(z_a) / (1 - norm.cdf(z_a))
            E_V_lt_b = mu - sigma * norm.pdf(z_b) / norm.cdf(z_b)

            profit_ask = (1 - pi) * (a - mu)       + pi * (a - E_V_gt_a)
            profit_bid = (1 - pi) * (mu - b)       + pi * (E_V_lt_b - b)

            return 0.5 * (profit_ask + profit_bid)   # MM trades ask & bid equally

        spread = float(fsolve(_expected_profit, initial_guess)[0])
        ask = mu + spread / 2
        bid = mu - spread / 2
        return bid, ask, spread

    # ---------- Distribución Exponencial ----------
    @staticmethod
    def bid_ask_exponential(
        lam: float = 0.5,
        pi: float = 0.3,
        initial_guess: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Equilibrium bid/ask for an exponentially-distributed fundamental value.

        Parameters
        ----------
        lam : float
            Rate parameter λ (mean = 1/λ).
        pi : float
            Probability that the counterparty is informed.
        initial_guess : float
            Initial guess for the non-linear solver.

        Returns
        -------
        bid : float
        ask : float
        spread : float  (ask − bid)
        """
        EV = 1.0 / lam

        def _expected_profit(S: float) -> float:
            a = EV + S / 2
            b = EV - S / 2

            # Conditional means for the exponential distribution
            E_V_gt_a = a + 1 / lam
            E_V_lt_b = (1 - np.exp(-lam * b) * (1 + lam * b)) / (lam * (1 - np.exp(-lam * b)))

            profit_ask = (1 - pi) * (a - EV) + pi * (a - E_V_gt_a)
            profit_bid = (1 - pi) * (EV - b) + pi * (E_V_lt_b - b)
            return 0.5 * (profit_ask + profit_bid)

        spread = float(fsolve(_expected_profit, initial_guess)[0])
        ask = EV + spread / 2
        bid = EV - spread / 2
        return bid, ask, spread
