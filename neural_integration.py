import numpy as np
import time
import torch
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from config import NUM_REALIZATIONS, SEED
from neural_model import train_neural_model, predict_with_neural_model, GeoCNN_LSTM, VariationalGeoCNN_LSTM

# Seed para reprodutibilidade
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def prepare_data_for_neural_model(X, n_derived_features):
    """
    Prepara os dados para o modelo neural
    
    Args:
        X: Features (com características derivadas)
        n_derived_features: Número de características derivadas
    
    Returns:
        X_processed: Dados processados prontos para o modelo neural
    """
    # Separar características originais e derivadas
    X_original = X[:, :-n_derived_features]
    X_derived = X[:, -n_derived_features:]
    
    # Normalizar características originais
    scaler = StandardScaler()
    X_original_scaled = scaler.fit_transform(X_original)
    
    # Combinar de volta
    X_processed = np.hstack((X_original_scaled, X_derived))
    
    return X_processed, scaler

def train_neural_ensemble(X_train, y_train, X_val, y_val, n_derived_features, 
                          model_type='variational', ensemble_size=3, 
                          save_path='neural_models'):
    """
    Treina um ensemble de modelos neurais para previsão geológica
    
    Args:
        X_train: Features de treinamento
        y_train: Alvos de treinamento
        X_val: Features de validação 
        y_val: Alvos de validação
        n_derived_features: Número de características derivadas
        model_type: 'deterministic' ou 'variational'
        ensemble_size: Número de modelos no ensemble
        save_path: Caminho para salvar os modelos
    
    Returns:
        Lista de modelos treinados e scaler
    """
    print(f"Treinando ensemble de {ensemble_size} modelos neurais ({model_type})...")
    
    # Preparar dados
    X_train_processed, scaler = prepare_data_for_neural_model(X_train, n_derived_features)
    X_val_processed = scaler.transform(X_val[:, :-n_derived_features])
    X_val_processed = np.hstack((X_val_processed, X_val[:, -n_derived_features:]))
    
    # Configurações diferentes para cada modelo do ensemble
    configs = []
    for i in range(ensemble_size):
        # Variar hiperparâmetros para diversificar o ensemble
        # Converter tipos numpy para Python nativos para evitar erros
        config = {
            'hidden_size': int(np.random.choice([64, 128, 256])),         # Converter para int Python
            'num_lstm_layers': int(np.random.choice([1, 2, 3])),          # Converter para int Python
            'dropout': float(np.random.uniform(0.1, 0.4)),                # Converter para float Python
            'kernel_size': int(np.random.choice([3, 5, 7])),              # Converter para int Python
            'batch_size': int(np.random.choice([32, 64, 128])),           # Converter para int Python
            'lr': float(np.random.choice([0.001, 0.0005, 0.0001]))        # Converter para float Python
        }
        configs.append(config)
    
    # Treinar modelos
    models = []
    histories = []
    
    for i, config in enumerate(configs):
        print(f"\nTreinando modelo {i+1}/{ensemble_size} com configuração:")
        print(config)
        
        # Treinar modelo
        model, history = train_neural_model(
            X_train_processed, y_train, 
            X_val_processed, y_val,
            model_type=model_type,
            hidden_size=config['hidden_size'],
            num_lstm_layers=config['num_lstm_layers'],
            dropout=config['dropout'],
            kernel_size=config['kernel_size'],
            batch_size=config['batch_size'],
            lr=config['lr'],
            epochs=100,
            patience=10
        )
        
        models.append(model)
        histories.append(history)
        
        # Salvar modelo
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history,
            'model_type': model_type
        }, f"{save_path}/neural_model_{i}.pt")
    
    # Salvar scaler
    joblib.dump(scaler, f"{save_path}/scaler.pkl")
    
    # Salvar informações do ensemble
    ensemble_info = {
        'configs': configs,
        'model_type': model_type,
        'n_derived_features': n_derived_features,
        'ensemble_size': ensemble_size
    }
    joblib.dump(ensemble_info, f"{save_path}/ensemble_info.pkl")
    
    print(f"Ensemble de modelos neurais treinado e salvo em '{save_path}'")
    
    return models, scaler, ensemble_info

def predict_with_neural_ensemble(X, n_derived_features, models=None, ensemble_info=None, 
                                scaler=None, num_realizations=NUM_REALIZATIONS, 
                                load_path='neural_models'):
    """
    Gera previsões com ensemble de modelos neurais
    
    Args:
        X: Features
        n_derived_features: Número de características derivadas
        models: Lista de modelos (se já carregados)
        ensemble_info: Informações do ensemble (se já carregadas)
        scaler: Scaler para normalização (se já carregado)
        num_realizations: Número de realizações
        load_path: Caminho para carregar modelos
    
    Returns:
        Array de previsões com shape (n_samples, num_realizations, n_outputs)
    """
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar modelos e informações se não fornecidos
    if models is None or ensemble_info is None or scaler is None:
        # Carregar informações do ensemble
        ensemble_info = joblib.load(f"{load_path}/ensemble_info.pkl")
        model_type = ensemble_info['model_type']
        ensemble_size = ensemble_info['ensemble_size']
        
        # Carregar scaler
        scaler = joblib.load(f"{load_path}/scaler.pkl")
        
        # Inicializar modelos
        models = []
        for i in range(ensemble_size):
            # Carregar checkpoint
            checkpoint = torch.load(f"{load_path}/neural_model_{i}.pt", map_location=device)
            config = checkpoint['config']
            
            # Criar modelo
            input_size = X.shape[1]
            output_size = 300  # Assumindo saída de tamanho 300
            
            if model_type == 'deterministic':
                model = GeoCNN_LSTM(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=config['hidden_size'],
                    num_lstm_layers=config['num_lstm_layers'],
                    dropout=config['dropout'],
                    kernel_size=config['kernel_size']
                ).to(device)
            else:  # variational
                model = VariationalGeoCNN_LSTM(
                    input_size=input_size,
                    output_size=output_size,
                    hidden_size=config['hidden_size'],
                    num_lstm_layers=config['num_lstm_layers'],
                    dropout=config['dropout'],
                    kernel_size=config['kernel_size']
                ).to(device)
            
            # Carregar pesos
            model.load_state_dict(checkpoint['model_state_dict'])
            models.append(model)
    
    # Modelo type
    model_type = ensemble_info['model_type']
    
    # Preparar dados
    X_original = X[:, :-n_derived_features]
    X_derived = X[:, -n_derived_features:]
    X_original_scaled = scaler.transform(X_original)
    X_processed = np.hstack((X_original_scaled, X_derived))
    
    # Fazer previsões com cada modelo do ensemble
    n_samples = X.shape[0]
    ensemble_size = len(models)
    # Vamos assumir que o tamanho da saída é 300
    output_size = 300
    
    # Base predictions - Média dos modelos determinísticos ou média das médias dos modelos variacionais
    base_predictions = np.zeros((n_samples, output_size))
    
    # Converter para tensor
    X_tensor = torch.FloatTensor(X_processed).to(device)
    
    # Fazer previsões com cada modelo
    print("Gerando previsões com o ensemble neural...")
    for model in models:
        model.eval()
        with torch.no_grad():
            if model_type == 'deterministic':
                pred = model(X_tensor).cpu().numpy()
                base_predictions += pred
            else:  # variational
                mu, _ = model.encode(X_tensor)
                pred = mu.cpu().numpy()
                base_predictions += pred
    
    # Calcular média das previsões
    base_predictions /= ensemble_size
    
    # Para modelos variacionais, geramos realizações
    if model_type == 'variational':
        # Precisamos gerar num_realizations para cada amostra
        realizations = np.zeros((n_samples, num_realizations, output_size))
        
        # Primeira realização é a previsão base (média dos modelos)
        realizations[:, 0, :] = base_predictions
        
        # Gerar realizações restantes
        reals_per_model = (num_realizations - 1) // ensemble_size + 1
        r_idx = 1
        
        for model_idx, model in enumerate(models):
            if r_idx >= num_realizations:
                break
                
            # Número de realizações para este modelo
            n_reals = min(reals_per_model, num_realizations - r_idx)
            
            with torch.no_grad():
                # Obter parâmetros da distribuição
                mu, logvar = model.encode(X_tensor)
                std = torch.exp(0.5 * logvar)
                
                # Gerar realizações
                for i in range(n_reals):
                    # Amostra da distribuição
                    eps = torch.randn_like(std)
                    sample = mu + eps * std
                    
                    # Adicionar à matriz de realizações
                    realizations[:, r_idx, :] = sample.cpu().numpy()
                    r_idx += 1
    else:
        # Para modelos determinísticos, usamos abordagem similar ao pipeline original
        # mas com base na previsão neural
        # Esta parte será melhorada integrando com as estratégias originais de geração
        realizations = np.zeros((n_samples, num_realizations, output_size))
        realizations[:, 0, :] = base_predictions
        
        # Gerar realizações simples com ruído para teste
        for r in range(1, num_realizations):
            noise_scale = 0.1 * r / num_realizations
            realizations[:, r, :] = base_predictions + np.random.normal(0, noise_scale, base_predictions.shape)
    
    return realizations

def combine_neural_and_ensemble_predictions(neural_realizations, ensemble_realizations, 
                                           neural_weight=0.5, num_realizations=NUM_REALIZATIONS):
    """
    Combina previsões do modelo neural e do ensemble tradicional
    
    Args:
        neural_realizations: Realizações do modelo neural
        ensemble_realizations: Realizações do ensemble tradicional
        neural_weight: Peso das previsões do modelo neural (0-1)
        num_realizations: Número de realizações no resultado
    
    Returns:
        Array de realizações combinadas
    """
    n_samples = neural_realizations.shape[0]
    output_size = neural_realizations.shape[2]
    
    # Ajustar peso do ensemble tradicional
    ensemble_weight = 1.0 - neural_weight
    
    # Inicializar realizações combinadas
    combined_realizations = np.zeros((n_samples, num_realizations, output_size))
    
    # Primeira realização: média ponderada das primeiras realizações
    combined_realizations[:, 0, :] = (
        neural_weight * neural_realizations[:, 0, :] + 
        ensemble_weight * ensemble_realizations[:, 0, :]
    )
    
    # Estratégia para realizações restantes: intercalar e ponderar
    neural_r = 1
    ensemble_r = 1
    
    for r in range(1, num_realizations):
        if r % 2 == 1 and neural_r < neural_realizations.shape[1]:
            # Usar uma realização do modelo neural com peso ajustado
            combined_realizations[:, r, :] = (
                neural_weight * neural_realizations[:, neural_r, :] + 
                ensemble_weight * ensemble_realizations[:, 0, :]  # Ancorar na previsão base do ensemble
            )
            neural_r += 1
        elif ensemble_r < ensemble_realizations.shape[1]:
            # Usar uma realização do ensemble tradicional com peso ajustado
            combined_realizations[:, r, :] = (
                neural_weight * neural_realizations[:, 0, :] +  # Ancorar na previsão base neural
                ensemble_weight * ensemble_realizations[:, ensemble_r, :]
            )
            ensemble_r += 1
        else:
            # Caso de fallback: usar o que estiver disponível
            if neural_r < neural_realizations.shape[1]:
                combined_realizations[:, r, :] = neural_realizations[:, neural_r, :]
                neural_r += 1
            elif ensemble_r < ensemble_realizations.shape[1]:
                combined_realizations[:, r, :] = ensemble_realizations[:, ensemble_r, :]
                ensemble_r += 1
            else:
                # Reuso circular de realizações
                neural_idx = r % neural_realizations.shape[1]
                ensemble_idx = r % ensemble_realizations.shape[1]
                combined_realizations[:, r, :] = (
                    neural_weight * neural_realizations[:, neural_idx, :] + 
                    ensemble_weight * ensemble_realizations[:, ensemble_idx, :]
                )
    
    return combined_realizations


def evaluate_neural_realizations(y_val, neural_realizations, ensemble_realizations=None, 
                                calculate_nll_fn=None, calibration_fn=None):
    """
    Avalia as realizações geradas pelo modelo neural
    
    Args:
        y_val: Valores verdadeiros
        neural_realizations: Realizações do modelo neural
        ensemble_realizations: Realizações do ensemble tradicional (opcional)
        calculate_nll_fn: Função para calcular NLL do módulo de avaliação original
        calibration_fn: Função para calibração do módulo original
        
    Returns:
        Dicionário com métricas de avaliação
    """
    # Importar apenas se necessário, para evitar dependências circulares
    if calculate_nll_fn is None:
        from utils.evaluation import calculate_nll_loss as calculate_nll_fn
    
    if calibration_fn is None:
        from enhanced_calibration import optimize_region_specific_calibration as calibration_fn
    
    # Calcular NLL para realizações neurais
    neural_nll = calculate_nll_fn(y_val, neural_realizations)
    print(f"NLL do modelo neural: {neural_nll:.2f}")
    
    # Comparar com ensemble tradicional, se fornecido
    if ensemble_realizations is not None:
        ensemble_nll = calculate_nll_fn(y_val, ensemble_realizations)
        print(f"NLL do ensemble tradicional: {ensemble_nll:.2f}")
        
        # Tentar diferentes pesos para combinação
        weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        nll_by_weight = {}
        
        for w in weights:
            combined = combine_neural_and_ensemble_predictions(
                neural_realizations, ensemble_realizations, neural_weight=w
            )
            combined_nll = calculate_nll_fn(y_val, combined)
            nll_by_weight[w] = combined_nll
            print(f"NLL combinado (peso neural={w}): {combined_nll:.2f}")
        
        # Encontrar melhor peso
        best_weight = min(nll_by_weight.items(), key=lambda x: abs(x[1] - (-69)))
        print(f"Melhor peso para modelo neural: {best_weight[0]}, NLL: {best_weight[1]:.2f}")
    
    # Tentar calibração usando o módulo existente
    print("\nCalibrando realizações do modelo neural...")
    calibration_params, calibrated_nll = calibration_fn(
        neural_realizations, y_val, target_nll=-69.0, max_iterations=15
    )
    
    print(f"NLL após calibração: {calibrated_nll:.2f}")
    
    # Construir resultado
    result = {
        'neural_nll': neural_nll,
        'calibrated_nll': calibrated_nll,
        'calibration_params': calibration_params
    }
    
    if ensemble_realizations is not None:
        result['ensemble_nll'] = ensemble_nll
        result['best_neural_weight'] = best_weight[0]
        result['best_combined_nll'] = best_weight[1]
    
    return result


def integrate_neural_model_to_pipeline(X_train, y_train, X_val, y_val, n_derived_features,
                                      model_type='variational', ensemble_size=3, 
                                      neural_weight=0.5, train_new=True,
                                      existing_ensemble_fn=None, calculate_nll_fn=None,
                                      calibration_fn=None, save_path='neural_models'):
    """
    Integra o modelo neural ao pipeline existente
    
    Args:
        X_train: Features de treinamento
        y_train: Alvos de treinamento
        X_val: Features de validação
        y_val: Alvos de validação
        n_derived_features: Número de características derivadas
        model_type: 'deterministic' ou 'variational'
        ensemble_size: Número de modelos no ensemble neural
        neural_weight: Peso inicial para o modelo neural na combinação
        train_new: Se deve treinar um novo modelo ou carregar um existente
        existing_ensemble_fn: Função para gerar previsões com ensemble existente
        calculate_nll_fn: Função para calcular NLL
        calibration_fn: Função para calibração
        save_path: Caminho para salvar/carregar modelos
    
    Returns:
        Dicionário com modelos, parâmetros e avaliações
    """
    start_time = time.time()

    # Treinar ou carregar modelos neurais
    if train_new:
        # Treinar novo ensemble neural
        neural_models, scaler, ensemble_info = train_neural_ensemble(
            X_train, y_train, X_val, y_val, n_derived_features,
            model_type=model_type, ensemble_size=ensemble_size,
            save_path=save_path
        )
    else:
        # Carregar modelos existentes
        neural_models, scaler, ensemble_info = None, None, None
    
    # Gerar realizações com modelo neural
    neural_realizations = predict_with_neural_ensemble(
        X_val, n_derived_features, models=neural_models,
        ensemble_info=ensemble_info, scaler=scaler,
        num_realizations=NUM_REALIZATIONS, load_path=save_path
    )
    
    # Gerar realizações com ensemble tradicional, se disponível
    if existing_ensemble_fn is not None:
        # Importar apenas se necessário, para evitar dependências circulares
        if existing_ensemble_fn is None:
            from models.ensemble import predict_with_advanced_ensemble
            existing_ensemble_fn = predict_with_advanced_ensemble
        
        # Obter realizações do ensemble tradicional
        # Aqui assumimos que esta função já foi definida em outro lugar do código
        ensemble_base_predictions = existing_ensemble_fn(X_val)
        
        # Gerar realizações completas (esta chamada depende da implementação existente)
        from enhanced_generator import generate_enhanced_realizations
        ensemble_realizations = generate_enhanced_realizations(
            X_val, ensemble_base_predictions, None, n_derived_features, NUM_REALIZATIONS
        )
    else:
        ensemble_realizations = None
    
    # Avaliar realizações
    evaluation = evaluate_neural_realizations(
        y_val, neural_realizations, ensemble_realizations,
        calculate_nll_fn, calibration_fn
    )
    
    # Se ensemble tradicional estiver disponível, tentar combinação
    if ensemble_realizations is not None:
        best_neural_weight = evaluation.get('best_neural_weight', neural_weight)
        
        # Combinar previsões
        combined_realizations = combine_neural_and_ensemble_predictions(
            neural_realizations, ensemble_realizations, 
            neural_weight=best_neural_weight, num_realizations=NUM_REALIZATIONS
        )
        
        # Aplicar calibração final na combinação
        if calibration_fn is not None:
            print("\nCalibrando realizações combinadas...")
            calibration_params, calibrated_nll = calibration_fn(
                combined_realizations, y_val, target_nll=-69.0, max_iterations=15
            )
            
            print(f"NLL combinado após calibração final: {calibrated_nll:.2f}")
            
            # Atualizar avaliação
            evaluation['combined_calibrated_nll'] = calibrated_nll
            evaluation['combined_calibration_params'] = calibration_params
            evaluation['final_realizations'] = combined_realizations
        else:
            evaluation['final_realizations'] = combined_realizations
    else:
        # Se não houver ensemble tradicional, usar apenas o neural calibrado
        from enhanced_calibration import apply_calibrated_scaling
        calibrated_realizations = apply_calibrated_scaling(
            neural_realizations, neural_realizations[:, 0, :], 
            evaluation['calibration_params']
        )
        evaluation['final_realizations'] = calibrated_realizations
    
    # Calcular tempo de execução
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTempo de execução: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Construir resultado
    result = {
        'models': neural_models,
        'scaler': scaler,
        'ensemble_info': ensemble_info,
        'evaluation': evaluation
    }
    
    return result