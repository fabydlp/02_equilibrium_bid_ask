## 02_equilibrium_bid_ask
Fabiana De la Peña & Santiago Figueiras

# Copeland–Galai Spread Calculator

Este repositorio contiene un módulo en Python para calcular el spread de equilibrio y los precios bid/ask según el modelo de Copeland & Galai (1983). El enfoque cubre dos distribuciones de valor fundamental: Normal (μ, σ²) y Exponencial (λ). La utilidad es ilustrar cómo, aun sin costos de transacción, el riesgo de operar contra inversores informados genera un spread que compensa al market maker.

---

## 1. Visión general del proyecto

- **Objetivo**:  
  Implementar, en un módulo fácilmente reutilizable, las fórmulas clave de Copeland & Galai para:
  1. Resolver el spread de equilibrio (S*) que anula el beneficio esperado del market maker frente a traders informados con probabilidad π.
  2. Calcular los precios bid (`b = V̄ − S*/2`) y ask (`a = V̄ + S*/2`), donde V̄ es el valor medio de la distribución del activo.
- **Distribuciones soportadas**:  
  - **Normal(μ, σ²)**  
  - **Exponencial(λ)**
- **Salida principal**:  
  Para cada caso, el método devuelve `(bid, ask, spread)`.

---

## 2. Fundamento teórico resumido

1. **Contexto Bagehot (1971)**  
   - En mercados financieros, el spread (ask − bid) no surge solo de costos explícitos: existe un componente informacional.  
   - En mercados con inversores con información privada, los market makers amplían el spread para protegerse del riesgo de vender caro o comprar barato.

2. **Modelo Copeland & Galai (1983)**  
   - Asume que el valor fundamental $V$ del activo sigue una distribución conocida (Normal o Exponencial).  
   - Los traders son informados con probabilidad $\pi$ y desinformados con probabilidad $1-\pi$.  
   - El market maker fija un spread $S$ y cotiza un precio ask  
     $$
       a = \mathbb{E}[V] + \frac{S}{2}
     $$  
     y un precio bid  
     $$
       b = \mathbb{E}[V] - \frac{S}{2}.
     $$  
   - El **beneficio esperado** del market maker, al operar al ask (vendiendo) y al bid (comprando), se anula en equilibrio:
     $$
       0 = \frac{1}{2} \Bigl[
         \underbrace{(1-\pi)\bigl(a - \mathbb{E}[V]\bigr) + \pi\bigl(a - \mathbb{E}[V \mid V > a]\bigr)}_{\text{beneficio en Ask}}
         \;+\;
         \underbrace{(1-\pi)\bigl(\mathbb{E}[V] - b\bigr) + \pi\bigl(\mathbb{E}[V \mid V < b] - b\bigr)}_{\text{beneficio en Bid}}
       \Bigr].
     $$
   - De esta ecuación se obtiene numéricamente $S^*$. Luego,
     $$
       \text{ask} = \mathbb{E}[V] + \frac{S^*}{2}, 
       \quad 
       \text{bid} = \mathbb{E}[V] - \frac{S^*}{2}.
     $$
   - **Distribución Normal**:  
     - $\displaystyle \mathbb{E}[V] = \mu.$  
     - $\displaystyle \mathbb{E}[V \mid V > a] = \mu + \sigma \,\frac{\varphi\!\bigl(\tfrac{a - \mu}{\sigma}\bigr)}{1 - \Phi\!\bigl(\tfrac{a - \mu}{\sigma}\bigr)}.$  
     - $\displaystyle \mathbb{E}[V \mid V < b] = \mu - \sigma \,\frac{\varphi\!\bigl(\tfrac{\mu - b}{\sigma}\bigr)}{\Phi\!\bigl(\tfrac{\mu - b}{\sigma}\bigr)}.$  
   - **Distribución Exponencial** (tasa $\lambda$):  
     - $\displaystyle \mathbb{E}[V] = \frac{1}{\lambda}.$  
     - $\displaystyle \mathbb{E}[V \mid V > a] = a + \frac{1}{\lambda}.$  
     - $\displaystyle \mathbb{E}[V \mid V < b] = \frac{\,1 - e^{-\lambda b}(1 + \lambda b)\,}{\lambda\bigl(1 - e^{-\lambda b}\bigr)}.$  


## 3. Estructura del paquete

02_equilibrium_bid_ask/
├── .gitignore
├── README.md
├── requirements.txt
└── technical_analysis/
├── init.py
├── utils.py
└── main.py


1. **`technical_analysis/__init__.py`**  
   - Expone la clase `CopelandGalaiCalc` para importarse directamente:
     ```python
     from .utils import CopelandGalaiCalc
     __all__ = ["CopelandGalaiCalc"]
     ```

2. **`technical_analysis/utils.py`**  
   - Contiene la definición de la clase `CopelandGalaiCalc` con métodos estáticos:
     - `bid_ask_normal(mu, sigma, pi, initial_guess) → (bid, ask, spread)`  
     - `bid_ask_exponential(lam, pi, initial_guess) → (bid, ask, spread)`

3. **`technical_analysis/main.py`**  
   - Script ejecutable para CLI que imprime los resultados de ejemplo:
     ```python
     from .utils import CopelandGalaiCalc

     def main():
         print("------- Normal Distribution -------")
         bid_n, ask_n, spread_n = CopelandGalaiCalc.bid_ask_normal(mu=102, sigma=7, pi=0.3)
         print(f"Bid: {bid_n:.4f}, Ask: {ask_n:.4f}, Spread: {spread_n:.4f}\n")
         
         print("----- Exponential Distribution -----")
         bid_e, ask_e, spread_e = CopelandGalaiCalc.bid_ask_exponential(lam=0.0075, pi=0.01)
         print(f"Bid: {bid_e:.4f}, Ask: {ask_e:.4f}, Spread: {spread_e:.4f}")

     if __name__ == "__main__":
         main()
     ```

---


## 4. Conclusion

En este proyecto hemos implementado de manera íntegra el modelo de Copeland & Galai para cuantificar el spread informacional en un mercado de valores. A través de un paquete de Python, se desarrollaron funciones que resuelven numéricamente el spread de equilibrio $S^*$ y, a partir de él, los precios bid y ask, tanto para una distribución normal del valor fundamental como para una distribución exponencial. Al explorar la dependencia de $S^*$ frente a la probabilidad de contrapartes informadas $\pi$ y el parámetro $\lambda$, se confirmó que un mayor riesgo informacional obliga a un spread más amplio y que la forma de la distribución incide directamente en el nivel de protección que debe exigir el market maker. En contraste con el método empírico de Roll, que extrae un spread implícito a partir de la covarianza de retornos, el enfoque de Copeland & Galai ofrece un marco teórico generado a partir de primero principios, permitiendo interpretar de manera explícita cómo cambian bid y ask cuando varían los supuestos de distribución y asimetría informacional. En definitiva, este trabajo no solo consolida los fundamentos matemáticos que explican la existencia de un spread aun en ausencia de costos de transacción, sino que provee una herramienta replicable y modular para analizar estrategias de creación de mercado y evaluar la liquidez desde una perspectiva estructural.
