import numpy as np
from scipy import stats

def regression_analysis(X, Y, alpha=0.05):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    n = len(X)
    
    # ממוצעים
    X_bar = np.mean(X)
    Y_bar = np.mean(Y)
    
    # שיפוע b_LS
    b_LS = np.sum((X - X_bar) * (Y - Y_bar)) / np.sum((X - X_bar)**2)
    
    # חיתוך a_LS
    a_LS = Y_bar - b_LS * X_bar
    
    # מקדם מתאם r_xy
    r_xy = np.sum((X - X_bar) * (Y - Y_bar)) / np.sqrt(np.sum((X - X_bar)**2) * np.sum((Y - Y_bar)**2))
    
    # R^2
    R2 = r_xy**2
    
    # ערכי Y חזויים
    Y_hat = a_LS + b_LS * X
    
    # סטיית תקן משוערת של השיפוע
    s2_hat = np.sum((Y - Y_hat)**2) / (n - 2)
    var_b = s2_hat / np.sum((X - X_bar)**2)
    se_b = np.sqrt(var_b)
    
    # סטטיסטי t לבדיקה H0: b=0
    t_stat = b_LS / se_b
    t_crit = stats.t.ppf(1 - alpha/2, df=n-2)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    
    return {
        "a_LS": a_LS,
        "b_LS": b_LS,
        "r_xy": r_xy,
        "R2": R2,
        "sigma^2_hat": s2_hat,
        "se_b": se_b,
        "t_stat": t_stat,
        "t_crit": t_crit,
        "p_value": p_value,
        "reject_H0": abs(t_stat) > t_crit
    }

# דוגמה לשימוש
# X = [1, 2, 3, 4, 5]
# Y = [2, 3, 4, 5, 7]

# results = regression_analysis(X, Y)
# for k, v in results.items():
#     print(f"{k}: {v}")