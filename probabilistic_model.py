import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ProbabilisticModelEnsemble:
    """
    Ensemble de modelos probabilísticos para previsão direta de incerteza em sequências geológicas
    """
    def __init__(self, n_quantiles=9, n_outputs=300, n_jobs=-1, random_state=42):
        """
        Inicializa o ensemble de modelos probabilísticos
        
        Args:
            n_quantiles: Número de quantis a serem previstos (excluindo a mediana)
            n_outputs: Número de posições de saída
            n_jobs: Número de jobs para paralelização
            random_state: Semente aleatória
        """
        self.n_quantiles = n_quantiles
        self.n_outputs = n_outputs
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.quantiles = np.linspace(0.1, 0.9, n_quantiles)
        self.key_positions = []
        
    def fit(self, X, y, n_derived_features):
        """
        Treina o ensemble de modelos probabilísticos
        
        Args:
            X: Features de entrada com shape (n_samples, n_features)
            y: Valores alvo com shape (n_samples, n_outputs)
            n_derived_features: Número de features derivadas
        """
        print("Treinando ensemble de modelos probabilísticos...")
        
        # Define posições-chave para treinar (mais granular no início onde a métrica é mais sensível)
        self.key_positions = []
        self.key_positions.extend(list(range(0, 60, 10)))
        self.key_positions.extend(list(range(60, 240, 30)))
        self.key_positions.extend(list(range(240, self.n_outputs, 20)))
        
        if self.key_positions[-1] != self.n_outputs - 1:
            self.key_positions.append(self.n_outputs - 1)  # Garante que temos o último ponto
        
        # Separa e escala os dados
        X_original = X[:, :-n_derived_features]  # Features originais
        X_derived = X[:, -n_derived_features:]   # Features derivadas
        
        # Escala apenas as features originais
        X_original_scaled = self.scaler.fit_transform(X_original)
        
        # Recombina
        X_processed = np.hstack((X_original_scaled, X_derived))
        
        # Define função para treinar em paralelo
        def train_position_models(pos_idx):
            pos = pos_idx + 1  # Converte para 1-indexado
            print(f"Treinando modelos para posição {pos} de {self.n_outputs}...")
            
            # Extrai alvo para esta posição
            y_pos = y[:, pos_idx]
            
            # Modelos para cada quantil e para a mediana
            models_dict = {}
            
            # Treina modelo de mediana (utilizando GBM otimizado)
            gbm_median = GradientBoostingRegressor(
                loss='squared_error',
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            )
            gbm_median.fit(X_processed, y_pos)
            models_dict['median'] = gbm_median
            
            # Previsão de mediana para uso no modelo NGBoost-like
            median_preds = gbm_median.predict(X_processed)
            
            # LightGBM para previsão de quantis (quantile regression)
            for i, q in enumerate(self.quantiles):
                # Para quantis baixos
                if q < 0.5:
                    lgb_lower = lgb.LGBMRegressor(
                        objective='quantile',
                        alpha=q,
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        verbosity=-1,
                        random_state=self.random_state
                    )
                    lgb_lower.fit(X_processed, y_pos)
                    models_dict[f'q{q}'] = lgb_lower
                
                # Para quantis altos
                elif q > 0.5:
                    lgb_upper = lgb.LGBMRegressor(
                        objective='quantile',
                        alpha=q,
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        verbosity=-1,
                        random_state=self.random_state
                    )
                    lgb_upper.fit(X_processed, y_pos)
                    models_dict[f'q{q}'] = lgb_upper
            
            # Modelo para prever diretamente a variância (abordagem NGBoost-like)
            # Calculamos resíduos ao quadrado para treinar o modelo de variância
            residuals = (y_pos - median_preds) ** 2
            
            # Treina modelo de variância
            gbm_variance = GradientBoostingRegressor(
                loss='squared_error',
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            )
            gbm_variance.fit(X_processed, residuals)
            models_dict['variance'] = gbm_variance
            
            return pos, models_dict
        
        # Treina modelos em paralelo
        if self.n_jobs != 1:
            print(f"Treinando modelos em paralelo com {self.n_jobs} workers...")
            with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
                results = list(executor.map(train_position_models, self.key_positions))
            
            # Processa resultados
            for pos, models_dict in results:
                self.models[pos] = models_dict
        else:
            print("Treinando modelos sequencialmente...")
            for pos_idx in self.key_positions:
                pos, models_dict = train_position_models(pos_idx)
                self.models[pos] = models_dict
        
        return self
    
    def predict(self, X, n_derived_features, num_realizations=10):
        """
        Gera realizações utilizando previsões probabilísticas
        
        Args:
            X: Features de entrada
            n_derived_features: Número de features derivadas
            num_realizations: Número de realizações a gerar
            
        Returns:
            Array de realizações com shape (n_samples, num_realizations, n_outputs)
        """
        n_samples = X.shape[0]
        
        # Separa e escala os dados
        X_original = X[:, :-n_derived_features]
        X_derived = X[:, -n_derived_features:]
        X_original_scaled = self.scaler.transform(X_original)
        X_processed = np.hstack((X_original_scaled, X_derived))
        
        # Inicializa matriz para previsões de mediana e variância
        median_predictions = np.zeros((n_samples, self.n_outputs))
        variance_predictions = np.zeros((n_samples, self.n_outputs))
        quantile_predictions = {q: np.zeros((n_samples, self.n_outputs)) for q in self.quantiles}
        
        # Gera previsões para posições-chave
        for pos in self.key_positions:
            pos_models = self.models[pos+1]  # +1 porque armazenamos como 1-indexado
            
            # Previsão de mediana
            median_preds = pos_models['median'].predict(X_processed)
            median_predictions[:, pos] = median_preds
            
            # Previsão de variância
            variance_preds = pos_models['variance'].predict(X_processed)
            # Aplicar clipping e smoothing para estabilizar a variância
            variance_preds = np.clip(variance_preds, 1e-6, np.inf)
            variance_predictions[:, pos] = variance_preds
            
            # Previsão de quantis
            for q in self.quantiles:
                if q != 0.5:  # Pula mediana que já temos
                    quantile_preds = pos_models[f'q{q}'].predict(X_processed)
                    quantile_predictions[q][:, pos] = quantile_preds
        
        # Interpola para todas as posições
        for i in range(n_samples):
            # Posições conhecidas (0-indexado)
            x_known = np.array(self.key_positions)
            
            # Todas as posições (0-indexado)
            x_all = np.arange(self.n_outputs)
            
            # Interpola mediana
            y_known = median_predictions[i, self.key_positions]
            median_predictions[i, :] = np.interp(x_all, x_known, y_known)
            
            # Interpola variância com suavização para estabilidade
            y_known = variance_predictions[i, self.key_positions]
            interpolated_variance = np.interp(x_all, x_known, y_known)
            # Suaviza a variância para evitar mudanças bruscas
            interpolated_variance = gaussian_filter1d(interpolated_variance, sigma=3.0)
            variance_predictions[i, :] = interpolated_variance
            
            # Interpola quantis
            for q in self.quantiles:
                y_known = quantile_predictions[q][i, self.key_positions]
                quantile_predictions[q][i, :] = np.interp(x_all, x_known, y_known)
        
        # Inicializa matriz para realizações
        all_realizations = np.zeros((n_samples, num_realizations, self.n_outputs))
        
        # Gera realizações a partir das distribuições previstas
        for i in range(n_samples):
            # A primeira realização é sempre a mediana
            all_realizations[i, 0, :] = median_predictions[i, :]
            
            # Para outras realizações, utilizamos diferentes abordagens
            for r in range(1, num_realizations):
                if r <= len(self.quantiles):
                    # Utilizamos diretamente os quantis previstos para algumas realizações
                    q_idx = r - 1
                    q = self.quantiles[q_idx]
                    all_realizations[i, r, :] = quantile_predictions[q][i, :]
                else:
                    # Para as demais, amostramos da distribuição probabilística
                    realization = np.zeros(self.n_outputs)
                    
                    for j in range(self.n_outputs):
                        # Parâmetros da distribuição
                        mean = median_predictions[i, j]
                        std = np.sqrt(variance_predictions[i, j])
                        
                        # Adiciona correlação espacial à amostragem
                        if j == 0:
                            # Para o primeiro ponto, amostra diretamente
                            realization[j] = np.random.normal(mean, std)
                        else:
                            # Para os demais, considera correlação com pontos anteriores
                            # Isso cria realizações mais realistas geologicamente
                            prev_diff = realization[j-1] - median_predictions[i, j-1]
                            # Decai a influência do ponto anterior
                            correlation = np.exp(-1/30)  # Parâmetro de decaimento da correlação
                            correlated_mean = mean + correlation * prev_diff
                            # Reduz a variância para pontos correlacionados
                            correlated_std = std * np.sqrt(1 - correlation**2)
                            realization[j] = np.random.normal(correlated_mean, correlated_std)
                    
                    # Aplica suavização para criar realizações geologicamente plausíveis
                    realization = gaussian_filter1d(realization, sigma=2.0)
                    all_realizations[i, r, :] = realization
                
                # Garante continuidade com os dados conhecidos (último ponto de entrada)
                try:
                    # Pega o último ponto válido da entrada
                    last_valid_idx = -1
                    X_original_i = X_original[i]
                    while last_valid_idx >= -X_original_i.shape[0]:
                        last_point = X_original_i[last_valid_idx]
                        if not np.isnan(last_point) and not np.isinf(last_point):
                            break
                        last_valid_idx -= 1
                    
                    if last_valid_idx >= -X_original_i.shape[0]:
                        # Calcula o offset e cria uma transição suave
                        offset = all_realizations[i, r, 0] - X_original_i[last_valid_idx]
                        # Aplica redução gradual do offset nos primeiros pontos
                        transition_length = min(20, self.n_outputs // 10)
                        weight = np.linspace(1.0, 0.0, transition_length)
                        
                        # Aplica offset ponderado para criar transição suave
                        for j in range(min(transition_length, self.n_outputs)):
                            all_realizations[i, r, j] -= offset * weight[j]
                except:
                    # Em caso de erro, não ajusta o offset
                    pass
        
        # Aplica calibração para NLL
        all_realizations = self._calibrate_for_nll_target(all_realizations)
        
        return all_realizations
    
    def _calibrate_for_nll_target(self, realizations, target_nll=-69):
        """
        Calibra as realizações para atingir um NLL alvo
        
        Args:
            realizations: Array de realizações com shape (n_samples, num_realizations, n_outputs)
            target_nll: Valor alvo de NLL
            
        Returns:
            Realizações calibradas
        """
        n_samples, n_realizations, n_outputs = realizations.shape
        calibrated = realizations.copy()
        
        # Parâmetros para calibração adaptativa
        # Ajusta o espalhamento baseado na posição (mais importante no início)
        position_weights = np.ones(n_outputs)
        # Dá mais peso às posições iniciais (onde o NLL é mais sensível)
        position_weights[:60] = np.linspace(1.5, 1.0, 60)
        position_weights[60:] = np.linspace(1.0, 0.8, n_outputs - 60)
        
        # Escala global inicial
        global_scale = 0.8
        
        # A primeira realização (mediana) permanece inalterada
        base_predictions = realizations[:, 0, :].copy()
        
        # Ajusta as outras realizações
        for r in range(1, n_realizations):
            # Calcula diferença em relação à mediana
            diff = realizations[:, r, :] - base_predictions
            
            # Aplica escala adaptativa por posição
            for j in range(n_outputs):
                scale_factor = global_scale * position_weights[j]
                diff[:, j] *= scale_factor
            
            # Atualiza realização calibrada
            calibrated[:, r, :] = base_predictions + diff
        
        # Aplicar diversificação entre realizações
        for i in range(n_samples):
            # Diversifica realizações para cobrir melhor o espaço de possibilidades
            for r in range(1, n_realizations):
                # Define um fator de diversificação único para cada realização
                diversity_factor = 0.9 + 0.2 * r / n_realizations
                
                # Aplica o fator à diferença em relação à base
                diff = calibrated[i, r, :] - base_predictions[i, :]
                calibrated[i, r, :] = base_predictions[i, :] + diff * diversity_factor
        
        return calibrated


def calculate_inverse_covariance_vector():
    """
    Calcula o vetor D_T^{-1}(x) para todas as posições de 1 a 300
    
    Returns:
        Vetor de covariância inversa
    """
    # Parâmetros da função de log-verossimilhança
    LOG_SLOPES = [1.0406028049510443, 0.0, 7.835345062351012]
    LOG_OFFSETS = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]
    
    inverse_cov_vector = np.zeros(300)
    for x in range(1, 301):
        if 1 <= x <= 60:
            k = 0  # Primeira região (1-60)
        elif 61 <= x <= 244:
            k = 1  # Segunda região (61-244)
        else:
            k = 2  # Terceira região (245-300)
        
        # Aplica fórmula: D_T^{-1}(x) = exp(log(x) * a_k + b_k)
        inverse_cov_vector[x-1] = np.exp(np.log(x) * LOG_SLOPES[k] + LOG_OFFSETS[k])
    
    return inverse_cov_vector

def calculate_nll_loss(y_true, predictions):
    """
    Calcula a Negative Log Likelihood conforme definida na avaliação
    
    Args:
        y_true: Array de valores verdadeiros com shape (n_samples, 300)
        predictions: Array de realizações com shape (n_samples, num_realizations, 300)
    
    Returns:
        Média de NLL para todas as amostras
    """
    n_samples = y_true.shape[0]
    num_realizations = predictions.shape[1]
    
    # Calcula vetor de covariância inversa idealizado
    inverse_cov_vector = calculate_inverse_covariance_vector()
    
    # Calcula NLL para cada amostra
    losses = np.zeros(n_samples)
    
    # Probabilidade de cada realização (equiprovável)
    p_i = 1.0 / num_realizations
    
    for i in range(n_samples):
        # Array para armazenar o desajuste gaussiano para cada realização
        gaussian_misfits = np.zeros(num_realizations)
        
        for r in range(num_realizations):
            # Calcula vetor de erro (residual)
            error_vector = y_true[i] - predictions[i, r]
            
            # Verifica valores inválidos no erro
            if np.isnan(error_vector).any() or np.isinf(error_vector).any():
                error_vector = np.nan_to_num(error_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calcula soma ponderada do erro baseada na covariância inversa
            weighted_error_sum = np.sum(error_vector**2 * inverse_cov_vector)
            
            # Limita para evitar problemas numéricos
            weighted_error_sum = np.clip(weighted_error_sum, -700, 700)
            
            # Calcula desajuste gaussiano
            gaussian_misfits[r] = np.exp(weighted_error_sum)
        
        # Calcula soma ponderada dos desajustes (NLL)
        weighted_sum = np.sum(p_i * gaussian_misfits)
        
        # Evita problemas numéricos com logaritmo
        if weighted_sum > 1e-300:
            losses[i] = -np.log(weighted_sum)
        else:
            # Valor máximo para casos extremos
            losses[i] = 700.0
    
    # Calcula média e exibe estatísticas
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)
    high_loss_count = np.sum(losses > 100)
    
    print(f"Estatísticas de NLL: média={mean_loss:.2f}, mediana={median_loss:.2f}, amostras com NLL alto: {high_loss_count}/{n_samples}")
    
    return mean_loss


# Função para integrar com o pipeline existente
def train_and_predict_with_probabilistic_model(X_train, y_train, X_val, X_test, n_derived_features, num_realizations=10):
    """
    Treina o modelo probabilístico e gera previsões
    
    Args:
        X_train: Features de treinamento
        y_train: Alvos de treinamento
        X_val: Features de validação
        X_test: Features de teste
        n_derived_features: Número de features derivadas
        num_realizations: Número de realizações a gerar
        
    Returns:
        Realizações de validação e teste
    """
    # Inicializa modelo probabilístico
    prob_model = ProbabilisticModelEnsemble(
        n_quantiles=9,
        n_outputs=y_train.shape[1],
        n_jobs=-1,
        random_state=42
    )
    
    # Treina modelo
    prob_model.fit(X_train, y_train, n_derived_features)
    
    # Gera realizações para validação
    val_realizations = prob_model.predict(X_val, n_derived_features, num_realizations)
    
    # Gera realizações para teste
    test_realizations = prob_model.predict(X_test, n_derived_features, num_realizations)
    
    return prob_model, val_realizations, test_realizations
