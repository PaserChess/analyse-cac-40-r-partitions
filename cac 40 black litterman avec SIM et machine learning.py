import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import itertools

# --- NOUVEAU : Import XGBoost avec gestion d'erreur ---
try:
    import xgboost as xgb
    print("✅ XGBoost détecté : Mode Turbo activé.")
    HAS_XGBOOST = True
except ImportError:
    print("⚠️ XGBoost non installé. Installation recommandée via 'pip install xgboost'.")
    print("👉 Fallback sur RandomForest (Scikit-Learn) pour l'instant.")
    from sklearn.ensemble import RandomForestRegressor
    HAS_XGBOOST = False

# --- 1. PARAMÈTRES --- meilleur : cap à 0.24 et min à 0.15
SEUIL_FILTRE = 0.15    # Filtre Conviction
WINDOW_YEARS = 3.5      # Fenêtre d'apprentissage
CONFIDENCE_LEVEL = 0.95 # VaR 95%
TARGET_BUDGET = 1000   
RISK_AVERSION = 2.5     # Standard pour les actions (Delta)
tau = 4               # Incertitude sur le Prior (Standard BL)
RISK_FREE_RATE = 0.025

print(f"1. CONFIG : BL + SIM + ML (XGBoost/RF) | Budget ~{TARGET_BUDGET}€")

# --- 2. DONNÉES HISTORIQUES ---
raw_history = {
    2010: ['AC.PA', 'AI.PA', 'AIR.PA','STLAP.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'OR.PA', 'MMB.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STMPA.PA', 'TTE.PA', 'URW.AS', 'VK.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2011: ['AC.PA', 'AI.PA', 'AIR.PA','STLAP.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'OR.PA', 'MC.PA', 'ML.PA', 'KN.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STMPA.PA', 'TTE.PA', 'URW.AS', 'VK.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2012: ['AC.PA', 'AI.PA', 'AIR.PA', 'STLAP.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STMPA.PA', 'TTE.PA', 'URW.AS', 'VK.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2013: ['AC.PA', 'AI.PA', 'AIR.PA','STLAP.PA',  'ALO.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SOLB.BR', 'STMPA.PA', 'TTE.PA', 'URW.AS', 'VK.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2014: ['AC.PA', 'AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SOLB.BR', 'TTE.PA', 'URW.AS', 'VK.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2015: ['AC.PA', 'AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SOLB.BR', 'TTE.PA', 'URW.AS', 'FR.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2016: ['AC.PA', 'AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'LI.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA',  'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SOLB.BR', 'TTE.PA', 'URW.AS', 'FR.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2017: ['AC.PA', 'AI.PA', 'AIR.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'LI.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'NOKIA.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SW.PA', 'SOLB.BR', 'TTE.PA', 'URW.AS', 'FR.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2018: ['AC.PA', 'AI.PA', 'AIR.PA', 'MT.AS', 'ATO.PA', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA', 'EL.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA','PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SW.PA', 'SOLB.BR', 'STMPA.PA', 'TTE.PA', 'URW.AS', 'FR.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2019: ['AC.PA', 'AI.PA', 'AIR.PA', 'MT.AS', 'ATO.PA', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SW.PA', 'STMPA.PA', 'TTE.PA', 'URW.AS', 'FR.PA', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2020: ['AC.PA', 'AI.PA', 'AIR.PA', 'MT.AS', 'ATO.PA', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'SW.PA', 'STMPA.PA', 'HO.PA', 'TTE.PA', 'URW.AS', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2021: ['AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'ATO.PA', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STMPA.PA', 'TEP.PA', 'HO.PA', 'TTE.PA', 'URW.AS', 'VIE.PA', 'DG.PA', 'VIV.PA', 'WLN.PA'],
    2022: ['AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'ERF.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA',  'STMPA.PA', 'TEP.PA', 'HO.PA', 'TTE.PA', 'URW.AS', 'VIE.PA', 'DG.PA', 'VIV.PA', 'WLN.PA'],
    2023: ['AI.PA', 'AIR.PA', 'ALO.PA','STLAP.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'ERF.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STLAP.PA', 'STMPA.PA', 'TEP.PA', 'HO.PA', 'TTE.PA', 'URW.AS', 'VIE.PA', 'DG.PA', 'VIV.PA', 'WLN.PA'],
    2024: ['AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'EDEN.PA', 'ENGI.PA', 'EL.PA', 'ERF.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STLAP.PA', 'STMPA.PA', 'TEP.PA', 'HO.PA', 'TTE.PA', 'URW.AS', 'VIE.PA', 'DG.PA', 'VIV.PA'],
    2025: ['AC.PA', 'AI.PA', 'AIR.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'BVI.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'EDEN.PA', 'ENGI.PA', 'EL.PA', 'ERF.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STLAP.PA', 'STMPA.PA', 'TEP.PA', 'HO.PA', 'TTE.PA', 'URW.AS', 'VIE.PA', 'DG.PA'],
    2026: ['AC.PA', 'AI.PA', 'AIR.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 'BVI.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'FGR.PA', 'ENGI.PA', 'EL.PA', 'ERF.PA', 'ENX.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STLAP.PA', 'STMPA.PA', 'HO.PA', 'TTE.PA', 'URW.AS', 'VIE.PA', 'DG.PA']
}
all_tickers_ever = sorted(list(set([t for yr in raw_history.values() for t in yr] + ['^FCHI'])))
print("2. TÉLÉCHARGEMENT DONNÉES...")
try:
    data = yf.download(all_tickers_ever, start="2005-01-01", auto_adjust=True)['Close'].ffill()
    prices_bench = data['^FCHI']
    last_date = data.index[-1]
    print(f"   Données OK jusqu'au {last_date.date()}")
except Exception as e:
    print(f"Erreur Download: {e}")
    exit()

# --- 3. FONCTIONS MATHÉMATIQUES ---

def get_covariance_sim(asset_returns, benchmark_returns):
    """
    Calcule la matrice de covariance via le SINGLE-INDEX MODEL.
    Cov = Beta * Beta' * Var(Market) + Diag(Residus)
    """
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 10: return None 
    
    y = asset_returns.loc[common_idx]
    x = benchmark_returns.loc[common_idx]
    
    var_mkt = x.var()
    if var_mkt < 1e-6: return asset_returns.cov() 
    
    covs_xy = y.apply(lambda col: col.cov(x))
    betas = covs_xy / var_mkt
    
    beta_outer = np.outer(betas, betas) * var_mkt
    
    var_assets = y.var()
    idio_vars = var_assets - (betas ** 2) * var_mkt
    idio_vars = np.maximum(idio_vars, 0) 
    
    sim_cov = beta_outer + np.diag(idio_vars)
    return pd.DataFrame(sim_cov, index=asset_returns.columns, columns=asset_returns.columns)

def predict_returns_ml(prices):
    """
    VERSION AMÉLIORÉE : Ajout de la distance à la moyenne mobile (Trend)
    """
    returns = prices.pct_change().dropna()
    predicted_means = {}
    
    for ticker in returns.columns:
        s = returns[ticker].copy()
        p = prices[ticker].copy() # On a besoin des prix pour les moyennes mobiles
        
        df_ml = pd.DataFrame({'t': s})
        
        # --- FEATURES ENGINEERING AMÉLIORÉ ---
        # 1. Momentum court terme
        df_ml['lag1'] = df_ml['t'].shift(1)       
        
        # 2. Volatilité
        df_ml['vol21'] = df_ml['t'].rolling(21).std().shift(1) 
        
        # 3. TENDANCE (Distance Prix vs Moyenne Mobile 50 jours)
        # Si > 0, l'action est en tendance haussière, le ML doit le savoir.
        sma50 = p.rolling(window=50).mean()
        dist_sma = (p / sma50) - 1
        # On aligne les index (car pct_change décale de 1)
        df_ml['dist_sma'] = dist_sma.reindex(df_ml.index).shift(1)

        df_ml = df_ml.dropna()
        
        if len(df_ml) < 60:
            predicted_means[ticker] = s.mean() * 252 
            continue
            
        X = df_ml[['lag1', 'vol21', 'dist_sma']]
        y = df_ml['t']
        
        if HAS_XGBOOST:
            # XGBoost un peu plus contraint pour éviter les folies
            model = xgb.XGBRegressor(  
                objective='reg:squarederror',
                n_estimators=220,      # Un peu plus d'arbres
                max_depth=6,          # Moins profond = Moins d'overfitting
                learning_rate=0.035,   # Apprentissage plus doux
                reg_alpha=0.4,        # Plus de régularisation L1
                n_jobs=-1,
                random_state=1
            )
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1, n_jobs=-1)
            
        model.fit(X, y)
        
        # Prédiction J+1
        last_sma = prices[ticker].rolling(50).mean().iloc[-1]
        last_dist = (prices[ticker].iloc[-1] / last_sma) - 1
        
        last_obs = pd.DataFrame([[
            s.iloc[-1],      
            s.iloc[-21:].std(),
            last_dist
        ]], columns=['lag1', 'vol21', 'dist_sma'])
        
        pred_daily = model.predict(last_obs)[0]
        
        # Annualisation
        pred_annual = pred_daily * 252
        hist_annual = s.mean() * 252
        
        # 50/50 : On fait confiance au ML mais on garde l'historique en ancre
        predicted_means[ticker] = (pred_annual * 0.5) + (hist_annual * 0.5)

    return pd.Series(predicted_means)

def optimize_black_litterman(prices_train, tau=tau): 
    """
    CORRECTION MAJEURE : Ajout de contraintes de poids max (Capping)
    """
    valid = prices_train.dropna(axis=1, how='any')
    if valid.shape[1] < 2: return None
    
    returns = valid.pct_change(fill_method=None).dropna()
    if returns.empty: return None
    returns = returns.loc[:, returns.var() > 1e-6]
    
    bench_slice = prices_bench.loc[returns.index]
    bench_rets = bench_slice.pct_change().dropna()
    
    if bench_rets.empty:
        cov_mat_annual = returns.cov() * 252
        risk_aversion = 2.5
    else:
        mkt_ret = bench_rets.mean() * 252
        mkt_var = bench_rets.var() * 252
        raw_lambda = (mkt_ret - RISK_FREE_RATE) / mkt_var 
        risk_aversion = np.clip(raw_lambda, 1.0, 10.0)
        cov_mat_annual = get_covariance_sim(returns, bench_rets) * 252
        if cov_mat_annual is None: cov_mat_annual = returns.cov() * 252

    mu_ml = predict_returns_ml(valid)
    mu_historical = mu_ml.loc[returns.columns]
    
    n_assets = len(mu_historical)
    jitter = 1e-5 * np.eye(n_assets)
    cov_mat_safe = cov_mat_annual + jitter
    
    # Calcul BL Classique (Identique à avant)
    weights_eq = np.array([1/n_assets] * n_assets).reshape(-1, 1)
    pi = risk_aversion * cov_mat_safe.dot(weights_eq)
    Q = mu_historical.values.reshape(-1, 1)
    P = np.identity(n_assets)
    omega = np.dot(np.dot(P, cov_mat_safe), P.T) * np.eye(n_assets)
    
    try:
        tau_cov_inv = np.linalg.pinv(tau * cov_mat_safe)
        omega_inv = np.linalg.pinv(omega)
        M1 = np.linalg.pinv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P))
        M2 = np.dot(tau_cov_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
        bl_returns = np.dot(M1, M2).flatten()
    except Exception:
        return None
    
    def neg_sharpe_bl(w):
        p_ret = np.sum(bl_returns * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat_safe, w))) 
        if p_vol < 1e-6: return 0 
        return - (p_ret - RISK_FREE_RATE) / p_vol

    # --- CHANGEMENT CRITIQUE ICI ---
    # Contrainte : Pas plus de 25% sur une seule action (0.25)
    # Cela force l'algo à prendre au moins 4 actions.
    MAX_WEIGHT = 0.24 
    bounds = tuple((0.0, MAX_WEIGHT) for _ in range(n_assets))
    
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    try:
        res = minimize(neg_sharpe_bl, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=cons)
        return pd.Series(res.x, index=returns.columns).sort_values(ascending=False)
    except:
        return None
    
def clean_portfolio(allocations, threshold):
    if allocations is None: return None
    clean = allocations[allocations >= threshold]
    if clean.empty: return allocations 
    return clean / clean.sum()

def calculate_risk_metrics(returns_series, confidence=0.95):
    """
    Calcule la VaR et l'Expected Shortfall (CVaR).
    """
    if returns_series.empty: return np.nan, np.nan
    
    var = np.percentile(returns_series, (1 - confidence) * 100)
    cvar = returns_series[returns_series <= var].mean()
    if np.isnan(cvar): cvar = var 
    
    return var, cvar

# --- 4. BACKTEST ENGINE ---
def run_scenario(scenario_id, name, start_year, end_year, use_filter):
    print(f"\n[{scenario_id}] {name}...")
    periods = []
    current_year = start_year
    while current_year <= end_year:
        periods.append((f"{current_year}-01-01", f"{current_year}-12-31"))
        current_year += 1
            
    cols = 3
    rows = math.ceil(len(periods) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.canvas.manager.set_window_title(name)
    fig.suptitle(name, fontsize=14, fontweight='bold', color='darkgreen')
    axs_flat = axes.flatten()
    plot_idx = 0
    
    cap_strat = 100.0
    cap_bench = 100.0
    risk_audit = [] 
    
    print(f"{'ANNÉE':<6} | {'BENCH':<8} | {'STRAT':<8} | {'VaR 95%':<8} | {'CVaR 95%':<9} | {'TOP POSITIONS':<30}")
    print("-" * 120)
    
    for start_test, end_test in periods:
        bench_slice = prices_bench.loc[start_test:end_test]
        if bench_slice.empty: continue
        
        b_rets = bench_slice.pct_change(fill_method=None).dropna()
        per_bench = (1 + b_rets).prod() - 1
        cap_bench *= (1 + per_bench)
        
        test_start_dt = pd.Timestamp(start_test)
        start_train = test_start_dt - pd.DateOffset(months=int(WINDOW_YEARS * 12))
        end_train = test_start_dt - pd.DateOffset(days=1)
        
        pool_year_int = int(start_test[:4]) - 1
        if pool_year_int not in raw_history: pool_year_int = 2026
        
        valid_tickers = sorted(list(set([t for t in raw_history[pool_year_int] if t in data.columns])))
        
        train_data = data.loc[start_train:end_train, valid_tickers]
        
        # Optimisation
        alloc = optimize_black_litterman(train_data)
        
        if use_filter:
            alloc = clean_portfolio(alloc, SEUIL_FILTRE)
        
        period_strat_curve = []
        period_bench_curve = []
        period_dates = []
        
        var_forecast = np.nan
        cvar_forecast = np.nan
        top_holdings_str = "CASH"
        
        if alloc is not None:
            top3 = alloc.head(3)
            top_holdings_str = ", ".join([f"{t.replace('.PA','')} {w*100:.0f}%" for t, w in top3.items()])
            
            # Calcul risques prévisionnels
            train_prices_alloc = train_data[alloc.index]
            train_rets_alloc = train_prices_alloc.pct_change(fill_method=None).dropna()
            
            train_port_rets = train_rets_alloc.dot(alloc) 
            
            var_forecast, cvar_forecast = calculate_risk_metrics(train_port_rets, CONFIDENCE_LEVEL)
            
            test_prices = data.loc[start_test:end_test, alloc.index]
            s_rets = test_prices.pct_change(fill_method=None).dropna()
            common_idx = s_rets.index.intersection(b_rets.index)
            
            if len(common_idx) > 0:
                s_rets = s_rets.loc[common_idx]
                b_rets_aligned = b_rets.loc[common_idx]
                
                daily_perf = s_rets.dot(alloc)
                
                worst_loss = daily_perf.min()
                breach_var = daily_perf[daily_perf < var_forecast]
                pct_breach = len(breach_var) / len(daily_perf) if len(daily_perf) > 0 else 0
                
                year_start = int(start_test[:4])
                if year_start >= 2021:
                    risk_audit.append({
                        "Period": f"{start_test[:4]}",
                        "Pred_VaR": var_forecast,
                        "Pred_CVaR": cvar_forecast,
                        "Real_Worst": worst_loss,
                        "Breach_Pct": pct_breach,
                        "Status": "FAIL" if pct_breach > (1-CONFIDENCE_LEVEL) else "OK"
                    })

                per_strat = (1 + daily_perf).prod() - 1
                cap_strat *= (1 + per_strat)
                
                curve_s = (1 + daily_perf).cumprod() * 100
                curve_b = (1 + b_rets_aligned).cumprod() * 100
                period_strat_curve = curve_s.tolist()
                period_bench_curve = curve_b.tolist()
                period_dates = common_idx

        per_bench_abs = per_bench
        per_strat_abs = per_strat if alloc is not None else 0
        sign = "✅" if per_strat_abs > per_bench_abs else "🔻"
        
        print(f"{start_test[:4]:<6} | {per_bench_abs*100:+.1f}%   | {per_strat_abs*100:+.1f}% {sign} | {var_forecast*100:.1f}%    | {cvar_forecast*100:.1f}%     | {top_holdings_str}")
        
        if plot_idx < len(axs_flat) and len(period_strat_curve) > 0:
            ax = axs_flat[plot_idx]
            ax.plot(period_dates, period_strat_curve, color='blue', linewidth=1.5)
            ax.plot(period_dates, period_bench_curve, color='gray', linestyle='--', alpha=0.7)
            col_tit = 'blue' if per_strat_abs > per_bench_abs else 'black'
            ax.set_title(f"{start_test[:4]} : {per_strat_abs*100:+.0f}% vs {per_bench_abs*100:+.0f}%", color=col_tit, fontsize=9, fontweight='bold')
            ax.xaxis.set_visible(False)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
    for j in range(plot_idx, len(axs_flat)): fig.delaxes(axs_flat[j])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("-" * 120)
    print(f"BILAN : Stratégie {cap_strat:.0f} € vs Bench {cap_bench:.0f} €")
    
    if len(risk_audit) > 0:
        print("\n>>> AUDIT DE RISQUE AVANCÉ (VaR vs CVaR)")
        print(f"{'Année':<8} | {'VaR (Seuil)':<12} | {'CVaR (Crash)':<13} | {'Pire Jour':<12} | {'Analyse'}")
        for r in risk_audit:
            cvar_breach = "⚠️ >CVaR" if r['Real_Worst'] < r['Pred_CVaR'] else "OK"
            print(f"{r['Period']:<8} | {r['Pred_VaR']*100:.2f}%      | {r['Pred_CVaR']*100:.2f}%       | {r['Real_Worst']*100:.2f}%      | {r['Status']} / {cvar_breach}")
            
    print("=" * 120)

# --- 5. OPTIMISEUR ENTIER ---
def smart_integer_optimizer(allocation, current_prices, target_budget):
    results = []
    tickers = allocation.index.tolist()
    target_weights = allocation.values
    ideal_amounts = target_weights * target_budget
    ideal_shares = ideal_amounts / current_prices[tickers].values
    
    ranges = []
    for share_count in ideal_shares:
        floor_s = math.floor(share_count)
        ceil_s = math.ceil(share_count)
        if floor_s == 0 and share_count > 0.1:
            ranges.append([1])
        else:
            ranges.append(list(set([floor_s, ceil_s])))
    
    if len(ranges) > 12:
         shares = np.round(ideal_shares)
         total_cost = np.sum(shares * current_prices[tickers].values)
         return {"shares": shares, "cost": total_cost, "score": 0}

    for combination in itertools.product(*ranges):
        shares = np.array(combination)
        total_cost = np.sum(shares * current_prices[tickers].values)
        if total_cost < target_budget * 0.8 or total_cost > target_budget * 1.2: continue
        
        real_weights = (shares * current_prices[tickers].values) / total_cost
        weight_error = np.sum(np.abs(real_weights - target_weights))
        budget_error = abs(total_cost - target_budget) / target_budget
        score = (2.0 * weight_error) + (1.0 * budget_error)
        
        results.append({"shares": shares, "cost": total_cost, "score": score, "real_weights": real_weights})
        
    if not results: return None
    return sorted(results, key=lambda x: x["score"])[0]

# --- 6. EXÉCUTION ---
scenarios_list = [
    {"id": 1, "years": (2011, 2020), "filter": True,  "desc": f"10 Ans | CONVICTION BL+SIM+XGBOOST (>12%)"},
    {"id": 2, "years": (2021, 2025), "filter": True,  "desc": f"5 Ans  | CONVICTION BL+SIM+XGBOOST (>12%)"},
]

print("3. LANCEMENT DES SIMULATIONS...")
for s in scenarios_list:
    run_scenario(s["id"], s["desc"], s["years"][0], s["years"][1], s["filter"])

# --- 7. PRÉVISIONS 2026 ---
print("\n" + "█"*80)
print(f"🔮 PRÉVISIONS 2026 : BLACK-LITTERMAN + SIM + XGBOOST (FINAL GOLD MASTER)")
print("█"*80)

end_train_26 = last_date
start_train_26 = last_date - pd.DateOffset(months=int(WINDOW_YEARS * 12))
tickers_2026 = [t for t in raw_history[2026] if t in data.columns]
train_data_26 = data.loc[start_train_26:end_train_26, tickers_2026]

# Appel Optimiseur BL
alloc_base = optimize_black_litterman(train_data_26)

if alloc_base is not None:
    budget = TARGET_BUDGET
    alloc_conviction = clean_portfolio(alloc_base, SEUIL_FILTRE)
    
    if alloc_conviction is not None:
        
        # --- DASHBOARD DE RISQUE 2026 ---
        train_prices_alloc = train_data_26[alloc_conviction.index]
        train_port_26 = train_prices_alloc.pct_change().dropna().dot(alloc_conviction.values)
        p_var_26, p_cvar_26 = calculate_risk_metrics(train_port_26, CONFIDENCE_LEVEL)
        p_vol_26 = train_port_26.std() * np.sqrt(252)
        p_ret_26 = train_port_26.mean() * 252
        
        print("\n📊 DASHBOARD DE RISQUE (Estimé sur 3.5 ans)")
        print("-" * 50)
        print(f"   Rendement Espéré (Hist) : {p_ret_26*100:.2f}%")
        print(f"   Volatilité Annuelle     : {p_vol_26*100:.2f}%")
        print(f"   Ratio de Sharpe (Est.)  : {(p_ret_26/p_vol_26):.2f}")
        print("-" * 50)
        print(f"   VaR 95% (Jour)          : {p_var_26*100:.2f}%  (Seuil de perte)")
        print(f"   CVaR 95% (Jour)         : {p_cvar_26*100:.2f}%  (Crash moyen)")
        print("-" * 50)
        
        print(f"\nA. PORTEFEUILLE THÉORIQUE")
        print("-" * 75)
        for t, w in alloc_conviction.items():
            price = data[t].iloc[-1]
            print(f"   👉 {t.replace('.PA',''):<10} | {w*100:.3f}%      | {budget*w:.0f} €          | {price:.2f} €")
    
        current_prices = data.iloc[-1][alloc_conviction.index]
        best_plan = smart_integer_optimizer(alloc_conviction, current_prices, TARGET_BUDGET)
        
        print("\n" + "█"*80)
        print(f"C. PLAN D'ACHAT OPTIMAL 2026 (Budget ~{TARGET_BUDGET}€)")
        print("█"*80)
        
        if best_plan:
            print(f"   Coût Total : {best_plan['cost']:.2f} €")
            print("-" * 80)
            print(f"{'ACTION':<10} | {'PRIX UNIT.':<12} | {'QTÉ':<10} | {'MONTANT':<12} | {'POIDS'}")
            print("-" * 80)
            
            tickers = alloc_conviction.index.tolist()
            shares = best_plan["shares"]
            
            for i, t in enumerate(tickers):
                n = int(shares[i])
                p = current_prices[t]
                amt = n * p
                w_real = amt / best_plan['cost']
                print(f"{t.replace('.PA',''):<10} | {p:<10.2f} € | {n:<10} | {amt:<10.0f} € | {w_real*100:.1f}%")
            print("-" * 80)
        else:
            print("Budget trop serré pour les prix unitaires.")
else:
    print("Erreur données.")

plt.show()