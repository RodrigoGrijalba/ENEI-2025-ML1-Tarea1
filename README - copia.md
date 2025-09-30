# Assignment: Linear Models, Regularization, and Model Selection on Real Data

**Deadline:** Sunday, October 5th, 2025, 23:59

**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

---

## Part A. Linear Regression From Scratch

1. **Dataset**
   Use the **California Housing dataset** (`from sklearn.datasets import fetch_california_housing`).

   * Create a hold-out test set.
   * Standardize features to zero mean and unit variance.
   * Predict the median house value (`MedHouseVal`) from the remaining features using `LinearRegression` from `sklearn.linear_model`.

2. **Closed-form OLS**

   * Derive and implement $\hat\beta = (X^\top X)^{-1}X^\top y$ using only `numpy`.
   * Report coefficients and intercept.
   * Plot predicted vs. true median house value on a held-out test set.

3. **Gradient Descent**

   * Implement gradient descent to minimize mean squared error.
   * Experiment with at least two learning rates; show cost vs. iteration curves.
   * Compare parameters and test error to the closed-form OLS.

---

## Part B. Scikit-learn Linear Models

4. **Baseline**

   * Use `LinearRegression` and confirm the coefficients match your OLS implementation.
   * Compute $R^2$ and mean squared error on the test set.

---

## Part C. Regularization and Hyperparameter Choice

5. **Ridge and Lasso**

   * Fit `Ridge` and `Lasso` regressions for $\lambda$ values logarithmically spaced between $10^{-3}$ and $10^{2}$.
   * Plot coefficient magnitude vs. $\lambda$ (regularization paths).
   * Comment on which features shrink to (or toward) zero and why.

6. **k-Fold Cross-Validation**

   * Use `KFold` with 5 folds and `cross_val_score` to select the best $\alpha$ for both Ridge and Lasso.
   * Alternatively, demonstrate the convenience of `RidgeCV` and `LassoCV`.
   * Compare cross-validated test errors.

7. **Feature Engineering & Multicollinearity**

  * Add polynomial features (degree 2) using `PolynomialFeatures`.
  * Re-run Ridge/Lasso and discuss how regularization copes with the enlarged feature space.

---

## Part D. Bike Rentals

8. **Alternative Dataset**

  * Use the **Bike Sharing Dataset** (available in the `data` folder).
  * Predict daily rentals (`cnt`); investigate seasonal effects.
  * Apply all the same steps as above.

---

### Deliverables
You must fork the [original repository](), and turn in a link to your groups repository with:

* A Jupyter notebook (in the `src` folder) with:

  * Your `numpy` implementations for OLS and gradient descent,
  * Plots: cost-function convergence, coefficient paths, predicted vs. actual values.
* A write-up in Markdown. Replace the contents of this file (`README.md`) with:
  
  * The names of your group's members,
  * Differences observed between OLS, Ridge, and Lasso,
  * Effect of learning rate on gradient descent,
  * How k-fold cross-validation influenced the choice of regularization strength.

