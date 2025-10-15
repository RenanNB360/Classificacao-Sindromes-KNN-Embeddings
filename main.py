import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from scipy.spatial.distance import cdist, euclidean, cosine
import collections
import sys
import logging


data_raw = 'mini_gm_public_v0.1.p'
range_k = range(1, 16)
splits = 10
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def processamnento_dados(data_raw):
    logging.info("--- Carregamento e Processamento dos Dados ---")
    with open(data_raw, 'rb') as f:
        data = pickle.load(f)

    flat_data = []
    for sind_id, subj in data.items():
        for subj_id, img in subj.items():
            for img_id, embed in img.items():
                flat_data.append({
                    'syndrome': sind_id,
                    'subject': subj_id,
                    'image': img_id,
                    'embedding': np.array(embed)
                })
    
    df = pd.DataFrame(flat_data)
    logging.info(f"Dados carregados com sucesso. Quantidade de registros: {len(df)}.")
    
    if df.isnull().sum().sum() > 0:
        logging.info("Foram encontrados valores ausentes.\n")
    else:
        logging.info("Não foram encontrados valores ausentes.\n")

    return df


def eda(df):
    logging.info("--- Análise Exploratória dos Dados ---")
    num_sind = df['syndrome'].nunique()
    logging.info(f"Quantidade de valores únicos por síndrome: {num_sind}.")

    imgs_sind = df.groupby('syndrome').size()
    logging.info("Imagens por síndrome (primeiros 5):")
    logging.info(f"{imgs_sind.nlargest(5)}\n")
    logging.info("Imagens por síndrome (últimos 5):")
    logging.info(f"{imgs_sind.nsmallest(5)}\n")

    logging.info("Distribuição da Síndrome:")
    cont_sind = df['syndrome'].value_counts(normalize=True) * 100
    logging.info(f"{cont_sind.round(2)}\n")

    if imgs_sind.std() > (imgs_sind.mean() / 2):
        logging.warning("Observação: Parece haver um desequilíbrio significativo no número de imagens por síndrome. Isso deve ser considerado durante a avaliação do modelo (por exemplo, usando validação cruzada estratificada).")
    else:
        logging.info("Observação: A distribuição das imagens entre as síndromes parece estar relativamente equilibrada.")
    
    logging.info(f"Dimensão das embeddings: {df['embedding'].iloc[0].shape[0]}.\n")


def visualizacao_tsne(df):
    logging.info("--- Visualização de dados com t-SNE ---")
    x = np.array(df['embedding'].tolist())
    y = df['syndrome'].astype('category').cat.codes

    logging.info("Rodando gráfico t-SNE")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    x_2d = tsne.fit_transform(x)
    logging.info("Gráfico de t-SNE concluído.\n")

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=np.unique(y), label='Síndrome')
    plt.title('Visualização t-SNE de Embeddings')
    plt.xlabel('Componente 1 do t-SNE')
    plt.ylabel('Componente 2 do t-SNE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('tsne.png')
    plt.show()


def roc_auc(y_true, y_pred, label):
    y_bin = (y_true == label).astype(int)
    if len(np.unique(y_bin)) < 2:
        logging.info("Somente uma classe presente")
        return np.nan, None, None
    fpr, tpr, _ = roc_curve(y_bin, y_pred)
    return auc(fpr, tpr), fpr, tpr



def topk_acc(y_true, probs, k=5):
    n, c = probs.shape
    if k > c: 
        k = c

    preds = np.argsort(probs, axis=1)[:, -k:]
    hits = (y_true == preds.T).any(axis=0)
    return hits.mean()


class KNN:
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X = None
        self.y = None
        self.classes = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)

    def probs(self, Xtest):
        if self.X is None:
            logging.info("Precisa rodar fit antes!")
            return None
        
        dists = cdist(Xtest, self.X, metric=self.metric)
        preds = np.zeros((Xtest.shape[0], len(self.classes)))

        for i, row in enumerate(dists):
            idx = np.argsort(row)[:self.k]
            labels = self.y[idx]
            for j, c in enumerate(self.classes):
                preds[i, j] = np.sum(labels == c) / self.k
        return preds

    def predict(self, Xtest):
        p = self.probs(Xtest)
        return self.classes[np.argmax(p, 1)]


def pipeline_classificacao(df):
    logging.info("--- Iniciando pipeline de classificação com KNN ---")

    X = np.array(df['embedding'].tolist())
    y_labels = df['syndrome'].to_numpy()
    unique_lbl = np.unique(y_labels)

    label_to_int = {label: i for i, label in enumerate(unique_lbl)}
    y_int = np.array([label_to_int[label] for label in y_labels])
    
    results = {
        'euclidean': {'auc': [], 'f1': [], 'top_k_acc': [], 'roc_curves': collections.defaultdict(list)},
        'cosine': {'auc': [], 'f1': [], 'top_k_acc': [], 'roc_curves': collections.defaultdict(list)}
    }
    best_k = {'euclidean': {'k': None, 'f1': -1}, 
              'cosine': {'k': None, 'f1': -1}}

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    for metric in ['euclidean', 'cosine']:
        logging.info(f"  Avaliando distância {metric}")
        
        f1_by_k = collections.defaultdict(list)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_int)):
            logging.info(f"  -> Fold {fold_idx + 1}/{splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_labels[train_idx], y_labels[test_idx]
            y_train_int, y_test_int = y_int[train_idx], y_int[test_idx]

            for k in range_k:
                knn = KNN(k=k, metric=metric)
                knn.fit(X_train, y_train)
                
                y_proba = knn.probs(X_test)
                y_pred = knn.predict(X_test)

                aucs = []
                roc_data = {}
                for lbl in unique_lbl:
                    auc_val, fpr, tpr = roc_auc(y_test, y_proba[:, label_to_int[lbl]], lbl)
                    if not np.isnan(auc_val):
                        aucs.append(auc_val)
                        roc_data[lbl] = (fpr, tpr)

                avg_auc = np.mean(aucs) if aucs else np.nan
                f1 = f1_score(y_test_int, [label_to_int[p] for p in y_pred], average='macro', zero_division=0)
                top_k = topk_acc(y_test_int, y_proba, k=5)

                f1_by_k[k].append(f1)

                if best_k[metric]['k'] is None or k == best_k[metric]['k']:
                    results[metric]['auc'].append(avg_auc)
                    results[metric]['f1'].append(f1)
                    results[metric]['top_k_acc'].append(top_k)
                    for lbl, (fpr, tpr) in roc_data.items():
                        results[metric]['roc_curves'][lbl].append((fpr, tpr))

        avg_f1_by_k = {k: np.mean(vals) for k, vals in f1_by_k.items()}
        chosen_k = max(avg_f1_by_k, key=avg_f1_by_k.get)
        best_k[metric]['k'] = chosen_k
        best_k[metric]['f1'] = avg_f1_by_k[chosen_k]

        logging.info(f"  Melhor k para {metric}: {chosen_k} (F1 médio: {avg_f1_by_k[chosen_k]:.4f})\n")

    return results, best_k, unique_lbl


def grafico_curvas_roc(results, unique_classes, metric_name):
    logging.info("--- Gerando gráfico de Curvas ROC ---")
    logging.info("Rodando gráfico Curvas ROC")
    plt.figure(figsize=(10, 8))
    plt.title(f'Curvas ROC Médias (Distância {'Euclidiana' if metric_name == 'euclidean' else 'do Cosseno'})')

    all_fpr, all_tpr, all_auc = [], [], []
    for class_label in unique_classes:
        fold_fprs = [fpr for fpr, _ in results[metric_name]['roc_curves'][class_label] if fpr is not None]
        fold_tprs = [tpr for _, tpr in results[metric_name]['roc_curves'][class_label] if tpr is not None]

        if not fold_fprs:
            continue

        base_fpr = np.linspace(0, 1, 101)
        interpolated_tprs = [
            np.interp(base_fpr, fpr, tpr) for fpr, tpr in zip(fold_fprs, fold_tprs)
        ]
        mean_tpr = np.mean(interpolated_tprs, axis=0)
        mean_auc = auc(base_fpr, mean_tpr)

        plt.plot(base_fpr, mean_tpr, label=f'Classe {class_label} (AUC = {mean_auc:.2f})')

        all_fpr.extend(base_fpr)
        all_tpr.extend(mean_tpr)
        all_auc.append(mean_auc)

    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.50)')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'grafico_{'Euclidiana' if metric_name == 'euclidean' else 'do Cosseno'}.png')
    plt.show()
    logging.info("Gráfico de Curvas ROC concluído.\n")


def tabela_sumarizacao(results, optimal_k_per_metric):
    logging.info("--- Resumo de Desempenho por Distância ---")
    resumo_metricas = []

    for dist in ['euclidean', 'cosine']:
        media_auc = np.mean(results[dist]['auc'])
        media_f1 = np.mean(results[dist]['f1'])
        media_top_k = np.mean(results[dist]['top_k_acc'])
        melhor_k = optimal_k_per_metric[dist]['k']

        resumo_metricas.append({
            'Tipo de Distância': dist.capitalize(),
            'K Ótimo': melhor_k,
            'AUC Médio': f'{media_auc:.4f}',
            'F1 Médio': f'{media_f1:.4f}',
            'Top-5 Acc Médio': f'{media_top_k:.4f}'
        })

    resumo_df = pd.DataFrame(resumo_metricas)
    logging.info(f"\n{resumo_df.to_string(index=False)}\n")

    logging.info("--- Comparação entre Distâncias ---")
    f1_euclidiano = np.mean(results['euclidean']['f1'])
    f1_cosseno = np.mean(results['cosine']['f1'])

    if f1_euclidiano > f1_cosseno:
        logging.info(f"Distância Euclidiana teve um desempenho ligeiramente superior (F1 Médio: {f1_euclidiano:.4f}) em comparação à Distância do Cosseno (F1 Médio: {f1_cosseno:.4f}).")
    elif f1_cosseno > f1_euclidiano:
        logging.info(f"Distância do Cosseno teve um desempenho ligeiramente superior (F1 Médio: {f1_cosseno:.4f}) em comparação à Distância Euclidiana (F1 Médio: {f1_euclidiano:.4f}).")
    else:
        logging.info("Ambas as distâncias apresentaram desempenho muito semelhante, sugerindo que os embeddings são robustos a diferentes métricas.")



if __name__ == "__main__":
    logging.info("--- Inciando o Pipeline ---\n")

    df = processamnento_dados(data_raw)
    eda(df)

    visualizacao_tsne(df)
    all_results, optimal_k_details, unique_syndromes = pipeline_classificacao(df)

    logging.info("--- Generating Evaluation Plots and Tables ---")
    grafico_curvas_roc(all_results, unique_syndromes, 'euclidean')
    grafico_curvas_roc(all_results, unique_syndromes, 'cosine')
    
    tabela_sumarizacao(all_results, optimal_k_details)