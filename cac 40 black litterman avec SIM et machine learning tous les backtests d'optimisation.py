import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import itertools
import time

# --- SETUP XGBOOST ---
try:
    import xgboost as xgb
    print("✅ XGBoost détecté : Mode Turbo activé.")
    HAS_XGBOOST = True
except ImportError:
    print("⚠️ XGBoost non installé. Arrêt critique.")
    exit()

# --- 1. PARAMÈTRES FIXES (ENVIRONNEMENT DE TEST) ---
WINDOW_YEARS = 3.5      
CONFIDENCE_LEVEL = 0.95 
TARGET_BUDGET = 1000    
RISK_FREE_RATE = 0.025

# CONTRAINTES FIXÉES (GOLD MASTER)
FIXED_CAP = 0.24
FIXED_MIN = 0.0   # ON RESTE SANS FILTRE POUR OPTIMISER LE MOTEUR PUR
FIXED_TAU = 1.0   # POINT DE DÉPART NEUTRE

print(f"1. CONFIG : ZOOMED GRID SEARCH (FINE TUNING) | Cap {FIXED_CAP} | Min {FIXED_MIN} | Tau {FIXED_TAU}")

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

# --- 3. FONCTIONS ---

def get_covariance_sim(asset_returns, benchmark_returns):
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

# --- HYPERPARAMETRES DYNAMIQUES ICI ---
def predict_returns_ml(prices, n_est, depth, lr, alpha):
    returns = prices.pct_change().dropna()
    predicted_means = {}
    
    for ticker in returns.columns:
        s = returns[ticker].copy()
        p = prices[ticker].copy()
        
        df_ml = pd.DataFrame({'t': s})
        df_ml['lag1'] = df_ml['t'].shift(1)        
        df_ml['vol21'] = df_ml['t'].rolling(21).std().shift(1) 
        sma50 = p.rolling(window=50).mean()
        dist_sma = (p / sma50) - 1
        df_ml['dist_sma'] = dist_sma.reindex(df_ml.index).shift(1)

        df_ml = df_ml.dropna()
        if len(df_ml) < 60:
            predicted_means[ticker] = s.mean() * 252 
            continue
            
        X = df_ml[['lag1', 'vol21', 'dist_sma']]
        y = df_ml['t']
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            reg_alpha=alpha,
            n_jobs=-1
        )
            
        model.fit(X, y)
        last_sma = prices[ticker].rolling(50).mean().iloc[-1]
        last_dist = (prices[ticker].iloc[-1] / last_sma) - 1
        last_obs = pd.DataFrame([[s.iloc[-1], s.iloc[-21:].std(), last_dist]], columns=['lag1', 'vol21', 'dist_sma'])
        pred_daily = model.predict(last_obs)[0]
        pred_annual = pred_daily * 252
        hist_annual = s.mean() * 252
        predicted_means[ticker] = (pred_annual * 0.5) + (hist_annual * 0.5)

    return pd.Series(predicted_means)

def optimize_black_litterman(prices_train, n_est, depth, lr, alpha): 
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

    # Passage des hyperparamètres XGBoost
    mu_ml = predict_returns_ml(valid, n_est, depth, lr, alpha)
    mu_historical = mu_ml.loc[returns.columns]
    
    n_assets = len(mu_historical)
    jitter = 1e-5 * np.eye(n_assets)
    cov_mat_safe = cov_mat_annual + jitter
    
    weights_eq = np.array([1/n_assets] * n_assets).reshape(-1, 1)
    pi = risk_aversion * cov_mat_safe.dot(weights_eq)
    Q = mu_historical.values.reshape(-1, 1)
    P = np.identity(n_assets)
    omega = np.dot(np.dot(P, cov_mat_safe), P.T) * np.eye(n_assets)
    
    try:
        # TAU FIXE A 1.0
        tau_cov_inv = np.linalg.pinv(FIXED_TAU * cov_mat_safe)
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

    # CAP FIXE A 0.24
    bounds = tuple((0.0, FIXED_CAP) for _ in range(n_assets))
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

# --- 4. BACKTEST ENGINE ADAPTÉ POUR RETOURNER UNE VALEUR ---
def run_scenario_silent(start_year, end_year, n_est, depth, lr, alpha):
    """
    Version silencieuse pour la grid search massive.
    Retourne la trésorerie finale.
    """
    periods = []
    current_year = start_year
    while current_year <= end_year:
        periods.append((f"{current_year}-01-01", f"{current_year}-12-31"))
        current_year += 1
            
    cap_strat = 100.0
    
    for start_test, end_test in periods:
        bench_slice = prices_bench.loc[start_test:end_test]
        if bench_slice.empty: continue
        
        test_start_dt = pd.Timestamp(start_test)
        start_train = test_start_dt - pd.DateOffset(months=int(WINDOW_YEARS * 12))
        end_train = test_start_dt - pd.DateOffset(days=1)
        
        pool_year_int = int(start_test[:4]) - 1
        if pool_year_int not in raw_history: pool_year_int = 2026
        
        valid_tickers = sorted(list(set([t for t in raw_history[pool_year_int] if t in data.columns])))
        train_data = data.loc[start_train:end_train, valid_tickers]
        
        alloc = optimize_black_litterman(train_data, n_est, depth, lr, alpha)
        
        # MIN FIXE A 0.0 (Désactivé)
        if alloc is not None:
             alloc = clean_portfolio(alloc, FIXED_MIN)
        
        if alloc is not None:
            test_prices = data.loc[start_test:end_test, alloc.index]
            s_rets = test_prices.pct_change(fill_method=None).dropna()
            
            # Alignement
            b_rets = bench_slice.pct_change(fill_method=None).dropna()
            common_idx = s_rets.index.intersection(b_rets.index)
            
            if len(common_idx) > 0:
                s_rets = s_rets.loc[common_idx]
                daily_perf = s_rets.dot(alloc)
                per_strat = (1 + daily_perf).prod() - 1
                cap_strat *= (1 + per_strat)

    return cap_strat

# --- 5. EXÉCUTION "ZOOM" GRID SEARCH ---
# Ranges "Zoomés" autour des champions
n_estimators_grid = [220, 230, 240]
max_depth_grid = [5, 6]
learning_rate_grid = [0.02,0.025, 0.03,0.035, 0.04]
reg_alpha_grid = [0.4]

param_combinations = list(itertools.product(n_estimators_grid, max_depth_grid, learning_rate_grid, reg_alpha_grid))
total_runs = len(param_combinations)

print(f"3. LANCEMENT DE LA GRID SEARCH 'ZOOM' (FINE TUNING)")
print(f"   Nombre de combinaisons : {total_runs}")
print(f"   Périodes : 2011-2020 (Valid) & 2021-2025 (Test)")
print("=" * 80)

results_store = []
best_perf_so_far = -999999
start_time = time.time()

for idx, (n_est, depth, lr, alpha) in enumerate(param_combinations):
    
    # 1. Backtest Historique (Validation)
    res_2010_2020 = run_scenario_silent(2011, 2020, n_est, depth, lr, alpha)
    
    # 2. Backtest Récent (Test)
    res_2021_2025 = run_scenario_silent(2021, 2025, n_est, depth, lr, alpha)
    
    # 3. Score Global (Somme des gains nets par rapport à la base 100)
    perf_hist = res_2010_2020 - 100
    perf_rec = res_2021_2025 - 100
    perf_total = perf_hist + perf_rec
    
    # Détection de record (Comparé à environ 508, le top précédent)
    is_new_record = perf_total > best_perf_so_far
    marker = ""
    if is_new_record:
        best_perf_so_far = perf_total
        marker = "👑 NEW RECORD"
    
    results_store.append({
        "params": f"Est:{n_est}|Dp:{depth}|LR:{lr}|Al:{alpha}",
        "n_est": n_est, "depth": depth, "lr": lr, "alpha": alpha,
        "res_10_20": res_2010_2020,
        "res_21_25": res_2021_2025,
        "perf_total": perf_total
    })
    
    # Affichage en temps réel
    print(f"[{idx+1}/{total_runs}] Est:{n_est} Dp:{depth} LR:{lr} Al:{alpha} | 10-20: {res_2010_2020:.0f}€ | 21-25: {res_2021_2025:.0f}€ | Tot: {perf_total:.0f} {marker}")

print("\n" + "="*80)
print("FIN DE LA GRID SEARCH - ANALYSE DES RÉSULTATS")
print("="*80)

# Conversion en DataFrame pour tri facile
df_res = pd.DataFrame(results_store)

print("\n🏆 TOP 10 : MEILLEURE PERFORMANCE HISTORIQUE (2011-2020)")
print(df_res.sort_values(by="res_10_20", ascending=False).head(10)[['params', 'res_10_20', 'res_21_25']].to_string(index=False))

print("\n🏆 TOP 10 : MEILLEURE PERFORMANCE RÉCENTE (2021-2025)")
print(df_res.sort_values(by="res_21_25", ascending=False).head(10)[['params', 'res_10_20', 'res_21_25']].to_string(index=False))

print("\n👑 TOP 10 : MEILLEURE PERFORMANCE GLOBALE (CUMULÉE)")
print(df_res.sort_values(by="perf_total", ascending=False).head(10)[['params', 'res_10_20', 'res_21_25', 'perf_total']].to_string(index=False))