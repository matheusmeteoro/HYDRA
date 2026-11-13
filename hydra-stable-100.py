import os
import sys
import argparse
import configparser
import time
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed

from scipy.stats import skew, kurtosis, shapiro, spearmanr
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.manifold import trustworthiness
from kneed import KneeLocator
from umap import UMAP

HYDRA_VERSION = "100"

# Esconde só o ruído conhecido; o resto dos avisos continua aparecendo
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*n_jobs.*')

def carregar_config(caminho):
    config = configparser.ConfigParser()
    if not os.path.exists(caminho):
        print(f"ERRO: Arquivo '{caminho}' não encontrado.")
        sys.exit(1)
    config.read(caminho)
    return config

def run_bootstrap_iteration(SEED, X_proc, n_samples, n_features, comps_main):
    """Executa uma iteração do Bootstrap para o PCA"""
    X_res = resample(X_proc, replace=True, n_samples=n_samples, random_state=SEED)
    pca_boot = PCA(svd_solver='full', n_components=n_features)
    pca_boot.fit(X_res)
    
    v_ratio = pca_boot.explained_variance_ratio_
    # ddof=1 para ser consistente com pca.explained_variance_ do sklearn
    total_var_res = np.var(X_res, axis=0, ddof=1).sum()
    mse_res = total_var_res - np.cumsum(pca_boot.explained_variance_)
    rmse = np.sqrt(np.maximum(mse_res, 0) / n_features)
    
    # Alinhamento: cada PC do bootstrap é casada com a PC principal de maior
    # sobreposição (trata troca de ordem) e depois o sinal é corrigido
    comps_boot = pca_boot.components_
    overlap = comps_boot @ comps_main.T
    lin, col = linear_sum_assignment(-np.abs(overlap))
    loadings = np.zeros_like(comps_boot)
    v_alinhada = np.zeros_like(v_ratio)
    for r, c in zip(lin, col):
        sinal = -1 if overlap[r, c] < 0 else 1
        loadings[c] = comps_boot[r] * sinal
        v_alinhada[c] = v_ratio[r]
    
    return v_ratio, rmse, loadings, v_alinhada

def main():
    start_mission_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.ini')
    args = parser.parse_args()
    cfg = carregar_config(args.config)

    # ================= CONFIGURAÇÕES =================
    INPUT_FILE = cfg['SETTINGS']['input_file']
    OUTPUT_PATH = cfg['SETTINGS']['output_path']
    INDEX_COLUMN = cfg['SETTINGS']['index_column'].strip("'\" ")
    FEATURES = [c.strip() for c in cfg['SETTINGS']['features'].replace('[', '').replace(']', '').split(',') if c.strip()]
    SEED = int(cfg['SETTINGS']['random_seed'])
    N_BOOTSTRAP = int(cfg['PCA']['n_bootstrap'])
    N_JOBS = int(cfg['PCA']['n_jobs'])
    RUN_UMAP = cfg.getboolean('UMAP', 'run_umap', fallback=True)
    RUN_GRID_UMAP = cfg.getboolean('UMAP', 'grid_search', fallback=True)
    
    TABLES_DIR = os.path.join(OUTPUT_PATH)
    os.makedirs(TABLES_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Código ANSI para Vermelho Brilhante e Reset
    RED = "\033[1;91m"
    RESET = "\033[0m"

    print(f"""{RED}

    ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗ 
    ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗
    ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║
    ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║
    ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║
    ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
    v{HYDRA_VERSION}
   """)
    
    # ================= CARREGAMENTO E INDEXAÇÃO =================
    
    print(f"{RED}Reading data and locking indices...")
    df_input = pd.read_csv(INPUT_FILE)
    df_input.columns = df_input.columns.astype(str).str.strip()
    
    if INDEX_COLUMN not in df_input.columns:
        print(f"ERRO FATAL: A coluna {INDEX_COLUMN} não existe no CSV.")
        sys.exit(1)

    # Colunas sem nome (header vazio vira 'Unnamed: N') são descartadas
    colunas_sem_nome = [c for c in df_input.columns if (c == '' or c.startswith('Unnamed:')) and c != INDEX_COLUMN]
    df_input = df_input.drop(columns=colunas_sem_nome)

    n_ids_duplicados = int(df_input[INDEX_COLUMN].duplicated().sum())
    if n_ids_duplicados > 0:
        print(f"ERRO FATAL: {n_ids_duplicados} IDs repetidos em '{INDEX_COLUMN}'. Remova as duplicatas.")
        sys.exit(1)
        
    df_tab = df_input.set_index(INDEX_COLUMN).copy()
    
    features_ausentes = [f for f in FEATURES if f not in df_tab.columns]
    features_presentes = [f for f in FEATURES if f in df_tab.columns]
    
    x_tab = df_tab[features_presentes].select_dtypes(include=[np.number])
    features_nao_numericas = [f for f in features_presentes if f not in x_tab.columns]
    features_presentes = list(x_tab.columns)
    metadados = [c for c in df_tab.columns if c not in features_presentes]

    if len(features_presentes) == 0:
        print("ERRO FATAL: nenhuma feature numérica encontrada.")
        sys.exit(1)

    # inf -> NaN, e remove linhas com NaN (o PCA não aceita)
    x_tab = x_tab.replace([np.inf, -np.inf], np.nan)
    nan_por_feature = x_tab.isna().sum()
    nan_por_feature = nan_por_feature[nan_por_feature > 0]
    n_total = len(x_tab)
    x_tab = x_tab.dropna()
    n_removidas = n_total - len(x_tab)
    
    if len(x_tab) < len(features_presentes):
        print(f"ERRO FATAL: só {len(x_tab)} galáxias válidas para {len(features_presentes)} features.")
        sys.exit(1)
    
    lower = x_tab.quantile(0.01)
    upper = x_tab.quantile(0.99)
    x_winsor = x_tab.clip(lower=lower, upper=upper, axis=1)
    
    scaler = RobustScaler(with_centering=True, with_scaling=True) 
    x_proc = scaler.fit_transform(x_winsor)
    n_samples, n_features = x_proc.shape

    # ================= PCA E BOOTSTRAP =================
    print(f"{RED}Executing PCA and Bootstrap...")
    pca_main = PCA(svd_solver='full', random_state=SEED)
    x_transform_main = pca_main.fit_transform(x_proc)
    comps_main = pca_main.components_
    var_acumulada_main = np.cumsum(pca_main.explained_variance_ratio_)
    
    total_var = np.var(x_proc, axis=0, ddof=1).sum()
    mse = total_var - np.cumsum(pca_main.explained_variance_)
    rmse_main = np.sqrt(np.maximum(mse, 0) / n_features)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_bootstrap_iteration)(SEED + i, x_proc, n_samples, n_features, comps_main) 
        for i in range(N_BOOTSTRAP)
    )
    
    boot_vars, boot_rmse, boot_loadings, boot_vars_alin = zip(*results)
    boot_vars, boot_rmse = np.array(boot_vars), np.array(boot_rmse)
    boot_loadings, boot_vars_alin = np.array(boot_loadings), np.array(boot_vars_alin)
    loadings_std = np.std(boot_loadings, axis=0)
    var_std = np.std(boot_vars, axis=0)

    # Escolha do K
    k_locator = KneeLocator(range(1, len(var_acumulada_main)+1), var_acumulada_main, curve='concave', direction='increasing')
    k_escolhido = int(k_locator.knee) if k_locator.knee else int(np.searchsorted(var_acumulada_main, 0.90) + 1)
    
    # Listas separadas: uma com TODAS as componentes, e outra só até o K.
    cols_pca_all = [f"PC{i+1}" for i in range(n_features)]
    cols_pca_k = [f"PC{i+1}" for i in range(k_escolhido)]
    
    # df_scores_all vai para o arquivo Master. df_scores_k é usado na estatística.
    df_scores_all = pd.DataFrame(x_transform_main, columns=cols_pca_all, index=x_tab.index)
    df_scores_k = df_scores_all[cols_pca_k]

    # ================= ESTATÍSTICA & DIAGNÓSTICOS =================
    print(f"{RED}Computing Correlations, Matrices, Colinearity and Importances...")
    
    corr_matrix_features = np.corrcoef(x_proc.T)
    # Regra "30" vale para o condition number da matriz de dados X = sqrt(cond da correlação)
    cond_number = np.sqrt(np.linalg.cond(corr_matrix_features))
    vif = np.diag(np.linalg.pinv(corr_matrix_features))
    pd.Series(vif, index=features_presentes, name='VIF').to_csv(os.path.join(TABLES_DIR, "stat_vif.csv"))

    # Correlacionando features originais vs as K componentes escolhidas
    df_combined = pd.concat([x_winsor, df_scores_k], axis=1)
    df_pearson = df_combined.corr(method='pearson').loc[features_presentes, cols_pca_k]
    df_spearman = df_combined.corr(method='spearman').loc[features_presentes, cols_pca_k]
    df_pearson.to_csv(os.path.join(TABLES_DIR, "stat_pearson.csv"))
    df_spearman.to_csv(os.path.join(TABLES_DIR, "stat_spearman.csv"))

    # Covariância e precisão das FEATURES (nos scores das PCs seriam diagonal, sem informação)
    df_cov = pd.DataFrame(x_proc, columns=features_presentes).cov()
    df_prec = pd.DataFrame(np.linalg.pinv(df_cov.values), index=features_presentes, columns=features_presentes)
    df_cov.to_csv(os.path.join(TABLES_DIR, "stat_covariance.csv"))
    df_prec.to_csv(os.path.join(TABLES_DIR, "stat_precision.csv"))

    dist_stats = []
    for pc in cols_pca_k:
        data_pc = df_scores_k[pc].values
        # Shapiro não é confiável com N > 5000: usa subamostra
        data_sw = data_pc if len(data_pc) <= 5000 else rng.choice(data_pc, 5000, replace=False)
        stat_sw, p_sw = shapiro(data_sw)
        dist_stats.append({
            'PC': pc, 'Mean': np.mean(data_pc), 'StdDev': np.std(data_pc, ddof=1),
            'Skewness': skew(data_pc), 'Kurtosis': kurtosis(data_pc),
            'Shapiro_W': stat_sw, 'Shapiro_p_value': p_sw, 'Shapiro_N': len(data_sw)
        })
    pd.DataFrame(dist_stats).to_csv(os.path.join(TABLES_DIR, "stat_pc_distributions.csv"), index=False)

    print(f"{RED}Calculating Global Importances and Errors...")
    # ---> IMPORTÂNCIAS GLOBAIS (restritas ao K escolhido) <---
    K = k_escolhido
    
    # A) Importância Absoluta: soma de |loading| nas K PCs, normalizada para 100%
    imp_absoluta = np.abs(comps_main[:K]).sum(axis=0)
    imp_absoluta_pct = 100.0 * imp_absoluta / imp_absoluta.sum()

    # B) Importância Ponderada: soma de loading^2 * variância_da_PC nas K PCs, normalizada para 100%
    imp_ponderada = ((comps_main[:K]**2) * pca_main.explained_variance_ratio_[:K, None]).sum(axis=0)
    imp_ponderada_pct = 100.0 * imp_ponderada / imp_ponderada.sum()

    # Erros: importância recalculada em CADA reamostragem; erro = std entre reamostragens
    bl = boot_loadings[:, :K, :]                 # (boot, K, features)
    bv = boot_vars_alin[:, :K]                   # (boot, K)
    abs_boot = np.abs(bl).sum(axis=1)
    abs_boot = 100.0 * abs_boot / abs_boot.sum(axis=1, keepdims=True)
    pond_boot = ((bl**2) * bv[:, :, None]).sum(axis=1)
    pond_boot = 100.0 * pond_boot / pond_boot.sum(axis=1, keepdims=True)
    err_absoluto_pct = np.std(abs_boot, axis=0)
    err_ponderado_pct = np.std(pond_boot, axis=0)

    df_importancia_pond = pd.Series(imp_ponderada_pct, index=features_presentes).sort_values(ascending=False)
    df_importancia_cumulativa = df_importancia_pond.cumsum()
    
    features_ordered = df_importancia_pond.index.tolist()
    err_pond_ordered = [err_ponderado_pct[features_presentes.index(f)] for f in features_ordered]
    imp_abs_ordered = [imp_absoluta_pct[features_presentes.index(f)] for f in features_ordered]
    err_abs_ordered = [err_absoluto_pct[features_presentes.index(f)] for f in features_ordered]

    df_global_ranking = pd.DataFrame({
        'Feature': features_ordered,
        'Peso_Ponderado_Pct': df_importancia_pond.values,
        'Erro_Ponderado_Pct': err_pond_ordered,
        'Peso_Absoluto_Pct': imp_abs_ordered,
        'Erro_Absoluto_Pct': err_abs_ordered,
        'Soma_Cumulativa_Pond_Pct': df_importancia_cumulativa.values
    })
    df_global_ranking.to_csv(os.path.join(TABLES_DIR, "stat_global_importance.csv"), index=False)

    # Distância de Mahalanobis nas K PCs (cada PC dividida por sqrt(autovalor))
    scores_k = df_scores_k.values
    centroid = np.mean(scores_k, axis=0)
    distancias_centroid = np.sqrt((((scores_k - centroid)**2) / pca_main.explained_variance_[:K]).sum(axis=1))
    top_outliers_idx = np.argsort(distancias_centroid)[-10:][::-1] 
    top_outliers = df_scores_k.index[top_outliers_idx]

    # ================= ARQUIVOS DE SUPORTE PARA PLOTAGEM =================
    print(f"{RED}Exporting Variances, RMSE and Loadings (ALL COMPONENTS)...")
    
    df_explained_variance = pd.DataFrame({
        'PC': cols_pca_all,
        'Explained_Variance_Ratio': pca_main.explained_variance_ratio_,
        'Explained_Variance_Error': var_std
    })
    df_explained_variance.to_csv(os.path.join(TABLES_DIR, "pca_explained_variance.csv"), index=False)

    df_rmse_export = pd.DataFrame({
        'PC': cols_pca_all,
        'RMSE': rmse_main,
        'RMSE_Std': np.std(boot_rmse, axis=0)
    })
    df_rmse_export.to_csv(os.path.join(TABLES_DIR, "pca_rmse_bruto.csv"), index=False)

    df_loadings_all = pd.DataFrame(comps_main.T, columns=cols_pca_all, index=features_presentes)
    df_loadings_err_all = pd.DataFrame(loadings_std.T, columns=[f"{c}_err" for c in cols_pca_all], index=features_presentes)
    df_loadings_full = pd.concat([df_loadings_all, df_loadings_err_all], axis=1)
    df_loadings_full.to_csv(os.path.join(TABLES_DIR, "pca_loadings_and_errors.csv"))
    

    # ================= UMAP 3D & TOPOLOGIA =================
    
    df_umap_final = pd.DataFrame(index=x_tab.index)
    df_umap_master = pd.DataFrame(index=x_tab.index)  # o que vai para o CSV master
    n_samp, n_trust = 0, 0
    if RUN_UMAP:
        print(f"{RED}Executing Manifold UMAP in 3D and Validating Topology...")
        
        # Métricas comparam o UMAP com o espaço COMPLETO (x_proc), não só com as K PCs
        n_samp = min(3000, n_samples)
        idx_rnd = rng.choice(n_samples, n_samp, replace=False)
        n_trust = max(1, min(15, (n_samp - 1) // 2))
        dados_originais_amostra = x_proc[idx_rnd]
        dist_orig = pdist(dados_originais_amostra)

        if RUN_GRID_UMAP:
            n_neighbors_list = [int(x.strip()) for x in cfg['UMAP'].get('n_neighbors_list', '15, 30').split(',')]
            min_dist_list = [float(x.strip()) for x in cfg['UMAP'].get('min_dist_list', '0.1, 0.3').split(',')]
            
            df_umap_all = pd.DataFrame(index=x_tab.index)
            resultados_metricas = []

            for n_n in n_neighbors_list:
                for m_d in min_dist_list:
                    tag = f"n{n_n}_d{m_d}"
                    reducer = UMAP(n_neighbors=n_n, min_dist=m_d, n_components=3, random_state=SEED)
                    umap_emb = reducer.fit_transform(x_transform_main[:, :k_escolhido])
                    
                    df_umap_all[f'UMAP1_{tag}'] = umap_emb[:, 0]
                    df_umap_all[f'UMAP2_{tag}'] = umap_emb[:, 1]
                    df_umap_all[f'UMAP3_{tag}'] = umap_emb[:, 2]

                    umap_amostra = umap_emb[idx_rnd]
                    
                    score_trust = trustworthiness(dados_originais_amostra, umap_amostra, n_neighbors=n_trust)
                    dist_emb = pdist(umap_amostra)
                    score_spearman_umap, _ = spearmanr(dist_orig, dist_emb)

                    resultados_metricas.append({
                        'n_neighbors': n_n,
                        'min_dist': m_d,
                        'Trustworthiness': score_trust,
                        'Spearman_Rho': score_spearman_umap,
                        'col_tag': tag
                    })
            
            df_umap_final = df_umap_all
            df_resultados = pd.DataFrame(resultados_metricas).sort_values(by='Trustworthiness', ascending=False)
            
            df_umap_final.to_csv(os.path.join(TABLES_DIR, 'coordenadas_todos_os_umaps.csv'))
            df_resultados.to_csv(os.path.join(TABLES_DIR, 'ranking_metricas_umap.csv'), index=False)

            # Master recebe só o melhor UMAP do grid (os outros ficam em coordenadas_todos_os_umaps.csv)
            tag_melhor = df_resultados.iloc[0]['col_tag']
            df_umap_master = df_umap_all[[f'UMAP{j}_{tag_melhor}' for j in (1, 2, 3)]].copy()
            df_umap_master.columns = ['UMAP1', 'UMAP2', 'UMAP3']

        else:
            n_n = int(cfg.get('UMAP', 'n_neighbors', fallback=30))
            m_d = float(cfg.get('UMAP', 'min_dist', fallback=0.1))
            
            reducer = UMAP(n_neighbors=n_n, min_dist=m_d, n_components=3, random_state=SEED)
            emb_for_metric = reducer.fit_transform(x_transform_main[:, :k_escolhido])
            
            df_umap_final = pd.DataFrame(emb_for_metric, columns=['UMAP1', 'UMAP2', 'UMAP3'], index=x_tab.index)
            df_umap_master = df_umap_final
            
            umap_amostra = emb_for_metric[idx_rnd]
            score_trust = trustworthiness(dados_originais_amostra, umap_amostra, n_neighbors=n_trust)
            dist_emb = pdist(umap_amostra)
            score_spearman_umap, _ = spearmanr(dist_orig, dist_emb)

            df_umap_final.to_csv(os.path.join(TABLES_DIR, 'coordenadas_umap_unico.csv'))
    else:
        print(f"{RED}Skipping UMAP calculation as per configuration...")

    # ================= CSV MASTER INTEGRADO =================
    print(f"{RED}Making final csv...")
    
    # Colunas originais + todas as PCs + UMAP (galáxias removidas por NaN ficam com PC/UMAP vazios)
    df_master = df_tab.join(df_scores_all).join(df_umap_master)
    caminho_master = os.path.join(TABLES_DIR, f"hydra{HYDRA_VERSION}.csv")
    df_master.to_csv(caminho_master)
    
    log_path = os.path.join(OUTPUT_PATH, "hydra_log.txt")
    
    var_tot_k = var_acumulada_main[k_escolhido-1] * 100
    # Erro da soma das K primeiras variâncias: std da SOMA em cada reamostragem
    var_err_k = np.std(np.sum(boot_vars[:, :k_escolhido], axis=1)) * 100
    rmse_pc1, rmse_final_k = rmse_main[0], rmse_main[k_escolhido-1]
    razao_var_erro = np.mean(pca_main.explained_variance_ratio_[:k_escolhido] / (var_std[:k_escolhido] + 1e-15))
    colinearidade_status = "ALTA (features redundantes)" if cond_number > 30 else "ok"
    n_pcs_total = len(pca_main.explained_variance_ratio_)
    n_pcs_90 = min(int(np.searchsorted(var_acumulada_main, 0.90) + 1), n_pcs_total)
    n_pcs_95 = min(int(np.searchsorted(var_acumulada_main, 0.95) + 1), n_pcs_total)

    with open(log_path, "w", encoding='utf-8') as f:
        f.write("="*110 + "\n")
        f.write(f" HYDRA v{HYDRA_VERSION} | LOG DA EXECUCAO | {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*110 + "\n\n")

        # === 1. DADOS ===
        f.write("1. DADOS\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Arquivo de entrada:  {INPUT_FILE}\n")
        f.write(f"Coluna de ID:        {INDEX_COLUMN}\n")
        f.write(f"N de galaxias:       {n_total} no arquivo | {n_samples} usadas | {n_removidas} removidas (NaN/inf)\n")
        if len(nan_por_feature) > 0:
            f.write(f"NaN/inf por feature: {', '.join(f'{k}={v}' for k, v in nan_por_feature.items())}\n")
        f.write(f"Features usadas ({len(features_presentes)}): {', '.join(features_presentes)}\n")
        f.write(f"Metadados mantidos ({len(metadados)}): {', '.join(metadados)}\n")
        if colunas_sem_nome:
            f.write(f"ATENCAO - colunas sem nome descartadas: {', '.join(colunas_sem_nome)}\n")
        if features_ausentes:
            f.write(f"ATENCAO - features pedidas que nao existem no arquivo: {', '.join(features_ausentes)}\n")
        if features_nao_numericas:
            f.write(f"ATENCAO - features nao numericas (ignoradas): {', '.join(features_nao_numericas)}\n")
        f.write(f"Condition number:    {cond_number:.2f} (sqrt do cond. da matriz de correlacao; colinearidade {colinearidade_status}; limite = 30)\n")
        f.write(f"CSV de saida:        {caminho_master}\n")
        f.write("VIF por feature (>10 = redundante): ")
        f.write(", ".join(f"{features_presentes[j]}={vif[j]:.2f}" for j in np.argsort(vif)[::-1]) + "\n")

        # === 2. IMPORTANCIA DAS FEATURES ===
        f.write(f"\n\n2. IMPORTANCIA DAS FEATURES (K = {K} PCs)\n")
        f.write("Peso pond. = soma(loading^2 * var. da PC) nas K PCs. Peso abs. = soma(|loading|) nas K PCs. Ambos normalizados p/ 100%.\n")
        f.write("Erro = desvio padrao entre reamostragens do bootstrap (importancia recalculada em cada uma).\n")
        f.write(f"{'-'*110}\n")
        f.write(f"{'#':<4} | {'FEATURE':<12} | {'PESO POND. (%)':<16} | {'ERRO POND.':<12} | {'PESO ABS. (%)':<16} | {'ERRO ABS.':<12} | {'CUMUL. POND. (%)'}\n")
        f.write(f"{'-'*110}\n")
        
        for i, row in df_global_ranking.iterrows():
            f.write(f"{i+1:<4} | {row['Feature']:<12} | {row['Peso_Ponderado_Pct']:<16.3f} | ± {row['Erro_Ponderado_Pct']:<9.3f} | {row['Peso_Absoluto_Pct']:<16.3f} | ± {row['Erro_Absoluto_Pct']:<9.3f} | {row['Soma_Cumulativa_Pond_Pct']:.3f}\n")

        f.write(f"\nTop 10 galaxias mais distantes do centro (Mahalanobis nas K PCs):\n")
        for i, gal in enumerate(top_outliers):
            dist = distancias_centroid[top_outliers_idx[i]]
            f.write(f"   {i+1:<2}. {str(gal):<15} dist = {dist:.2f}\n")

        # === 3. PCA ===
        f.write(f"\n\n3. PCA\n")
        f.write(f"{'-'*80}\n")
        f.write(f"PCs totais:            {n_pcs_total}\n")
        f.write(f"PCs retidas (K):       {k_escolhido} de {n_features}\n")
        f.write(f"PCs p/ 90% / 95%:      {n_pcs_90} / {n_pcs_95}\n")
        f.write(f"Variancia acumulada K: {var_tot_k:.2f}% (± {var_err_k:.2f}%, std bootstrap da soma das K variancias)\n")
        f.write(f"RMSE de reconstrucao:  {rmse_pc1:.4f} (1 PC) -> {rmse_final_k:.4f} (K PCs)\n")
        f.write(f"Var/Erro medio (K PCs): {razao_var_erro:.2f}\n")
        
        f.write(f"\n{'PC':<4} | {'VAR %':<8} | {'ERRO BOOT':<10} | {'CUMUL %':<8} | {'AUTOVALOR':<10} | {'VAR/ERRO':<9} | {'3 FEATURES MAIS PESADAS'}\n")
        f.write(f"{'-'*110}\n")
        for i in range(k_escolhido):
            v_pct, v_err = pca_main.explained_variance_ratio_[i] * 100, var_std[i] * 100
            c_pct = np.sum(pca_main.explained_variance_ratio_[:i+1]) * 100
            drivers = ", ".join([features_presentes[j] for j in np.argsort(np.abs(pca_main.components_[i]))[::-1][:3]])
            f.write(f"{i+1:<4} | {v_pct:<8.2f} | ±{v_err:<9.2f} | {c_pct:<8.2f} | {pca_main.explained_variance_[i]:<10.4f} | {v_pct/(v_err+1e-9):<9.1f} | {drivers}\n")

        # === 4. LOADINGS ===
        f.write(f"\n\n4. COMPOSICAO DAS PCs (loadings com |peso| >= 0.20, ± = erro bootstrap)\n")
        f.write(f"{'-'*80}\n")
        for i in range(k_escolhido):
            f.write(f" PC{i+1} =\n")
            params = sorted([(pca_main.components_[i][j], features_presentes[j], loadings_std[i][j]) for j in range(len(features_presentes))], key=lambda x: abs(x[0]), reverse=True)
            for w, n, e in params:
                if abs(w) >= 0.20:
                    f.write(f"      {'+' if w >= 0 else '-'} ({abs(w):.3f} ± {e:.3f}) * {n}\n")

        # === 5. UMAP ===
        f.write(f"\n\n5. UMAP (3D)\n")
        f.write(f"{'-'*80}\n")
        if RUN_UMAP:
            f.write(f"Entrada do UMAP: K = {k_escolhido} PCs. Metricas calculadas contra as {n_features} features (x_proc), em {n_samp} galaxias sorteadas.\n")
            if k_escolhido <= 3:
                f.write("ATENCAO - K <= 3: o UMAP vai de K dimensoes para 3, quase nao reduz nada.\n")
            f.write(f"Trustworthiness = preservacao da vizinhanca local (n_neighbors = {n_trust}; bom > 0.85).\n")
            f.write("Spearman = correlacao entre distancias originais e do UMAP (preservacao global).\n\n")
            
            if RUN_GRID_UMAP:
                f.write("Modo: grid search (CSV master recebe so o melhor; todos em coordenadas_todos_os_umaps.csv)\n\n")
                melhor_teste = df_resultados.iloc[0]
                f.write("Melhor combinacao:\n")
                f.write(f"  n_neighbors:     {int(melhor_teste['n_neighbors'])}\n")
                f.write(f"  min_dist:        {melhor_teste['min_dist']}\n")
                f.write(f"  Trustworthiness: {melhor_teste['Trustworthiness']:.4f}\n")
                f.write(f"  Spearman:        {melhor_teste['Spearman_Rho']:.4f}\n\n")

                f.write("Todos os testes:\n")
                f.write(f" {'n_neighbors':<12} | {'min_dist':<9} | {'Trustworthiness':<16} | {'Spearman'}\n")
                f.write(f" {'-'*55}\n")
                for _, row in df_resultados.iterrows():
                    f.write(f" {int(row['n_neighbors']):<12} | {row['min_dist']:<9.2f} | {row['Trustworthiness']:<16.4f} | {row['Spearman_Rho']:.4f}\n")
            else:
                f.write("Modo: execucao unica\n")
                f.write(f"n_neighbors = {n_n} | min_dist = {m_d}\n")
                f.write(f"Trustworthiness: {score_trust:.4f}\n")
                f.write(f"Spearman:        {score_spearman_umap:.4f}\n")
        else:
            f.write("UMAP desligado (run_umap=False).\n")

        # === 6. TEMPO ===
        seg = int(time.time() - start_mission_time)
        duracao = f"{seg // 3600:02d}:{(seg % 3600) // 60:02d}:{seg % 60:02d}"
        f.write(f"\n\n6. EXECUCAO\n")
        f.write(f"{'-'*50}\n")
        f.write(f"Tempo total: {duracao} | Threads: {N_JOBS if N_JOBS != -1 else 'todas'}\n")
        f.write("="*110 + "\n")

    print(f"\nProcessing time: {duracao}{RESET}")
    
if __name__ == "__main__":
    main()
