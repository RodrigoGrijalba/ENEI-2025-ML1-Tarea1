# Parte D — Modelos Lineales en Bike Rentals

Este apartado aplica los conceptos de regresión lineal, gradiente descendente y regularización al dataset de alquiler de bicicletas (`hour.csv`), recopilado en Washington D.C. durante 2011–2012.

---

## 1. Preprocesamiento de datos
- Fuente: archivo `hour.csv`, con registros horarios de demanda y condiciones climáticas.  
- Transformaciones aplicadas:
  - Conversión de `dteday` a tipo `datetime`.
  - Agregación a nivel diario (731 observaciones):
    - Suma: variables de conteo (`cnt`, `casual`, `registered`).
    - Promedio: variables climáticas (`temp`, `atemp`, `hum`, `windspeed`).
    - Moda: variables categóricas (`season`, `yr`, `mnth`, `weekday`, `holiday`, `workingday`, `weathersit`).  
- Resultado: dataset diario con variables representativas de clima, calendario y demanda.

---

## 2. Regresión OLS (desde cero)
- Implementación manual de la ecuación normal:  
  \[
  \hat{\beta} = (X^\top X)^{-1}X^\top y
  \]
- Resultados en test:
  - MSE ≈ 583,671  
  - R² ≈ 0.833
- Interpretación: la temperatura y el año incrementan la demanda, mientras que humedad, viento y clima adverso la reducen.

---

## 3. Gradiente Descendente (GD)
- Se implementó GD para minimizar el MSE.  
- Se probaron dos tasas de aprendizaje: η = 0.001 y η = 0.01.  
- Ambos convergieron a soluciones cercanas a OLS.  
- Mejor resultado (η=0.001):
  - MSE ≈ 583,554  
  - R² ≈ 0.833

Las curvas de costo muestran convergencia estable sin oscilaciones.

---

## 4. Baseline con `scikit-learn`
- Se entrenó `LinearRegression` con los mismos datos estandarizados.  
- Los coeficientes y métricas coincidieron exactamente con OLS manual.  
- Confirma la correcta implementación de la ecuación normal y la equivalencia entre ambas aproximaciones.

---

## 5. Regularización (Ridge y Lasso)
- Se evaluaron Ridge y Lasso con validación cruzada (10 folds).  
- Mejores hiperparámetros:
  - Ridge: α ≈ 5.96  
  - Lasso: α ≈ 0.05
- Resultados en test:
  - Ridge → MSE ≈ 584,478, R² ≈ 0.832
  - Lasso → MSE ≈ 583,616, R² ≈ 0.833

Ambos alcanzan desempeño prácticamente idéntico a OLS, con ligeras diferencias:
- Ridge (L2): estabiliza coeficientes en presencia de colinealidad.  
- Lasso (L1): introduce esparsidad, eliminando predictores redundantes.  

---

## 6. Conclusión 
- El modelo lineal describe adecuadamente la demanda de bicicletas diarias (R² ≈ 0.83).  
- OLS, GD y `LinearRegression` producen resultados equivalentes.  
- Ridge y Lasso no mejoran sustancialmente el error en este dataset, pero ofrecen ventajas interpretativas:
  - Ridge: mayor estabilidad de coeficientes.  
  - Lasso: selección automática de variables.  

La Parte D nos confirma que la regularización es útil como herramienta de robustez, aunque en este caso no aporta mejoras significativas en el desempeño predictivo frente al modelo lineal base.
