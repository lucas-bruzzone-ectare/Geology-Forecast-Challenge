import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from config import TRAIN_PATH, TEST_PATH, NUM_REALIZATIONS, SEED
from utils.data_utils import load_data, prepare_data
from utils.evaluation import calculate_nll_loss
from neural_model import train_neural_model, GeoCNN_LSTM, VariationalGeoCNN_LSTM
from neural_integration import (
    prepare_data_for_neural_model,
    predict_with_neural_ensemble,
    combine_neural_and_ensemble_predictions
)

# Seed para reprodutibilidade
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def experiment_hyperparameter_search():
    """
    Experimento para busca de hiperparâmetros do modelo neural
    """
    print("Experimento: Busca de hiperparâmetros para modelo neural")
    
    # Carregar e preparar dados
    train_df, _, input_cols, output_cols = load_data(TRAIN_PATH)
    X, X_val, y, y_val, n_derived_features = prepare_data(train_df, input_cols, output_cols)
    
    # Dividir dados para treinamento e validação rápida
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
    
    # Preparar dados para modelo neural
    X_train_processed, scaler = prepare_data_for_neural_model(X_train, n_derived_features)
    X_test_processed = scaler.transform(X_test[:, :-n_derived_features])
    X_test_processed = np.hstack((X_test_processed, X_test[:, -n_derived_features:]))
    
    # Definir diferentes configurações
    configs = [
        # Variações de arquitetura
        {'model_type': 'deterministic', 'hidden_size': 128, 'num_lstm_layers': 2, 'kernel_size': 3, 'dropout': 0.2},
        {'model_type': 'variational', 'hidden_size': 128, 'num_lstm_layers': 2, 'kernel_size': 3, 'dropout': 0.2},
        
        # Variações de tamanho
        {'model_type': 'variational', 'hidden_size': 64, 'num_lstm_layers': 2, 'kernel_size': 3, 'dropout': 0.2},
        {'model_type': 'variational', 'hidden_size': 256, 'num_lstm_layers': 2, 'kernel_size': 3, 'dropout': 0.2},
        
        # Variações de profundidade
        {'model_type': 'variational', 'hidden_size': 128, 'num_lstm_layers': 1, 'kernel_size': 3, 'dropout': 0.2},
        {'model_type': 'variational', 'hidden_size': 128, 'num_lstm_layers': 3, 'kernel_size': 3, 'dropout': 0.2},
        
        # Variações de kernel
        {'model_type': 'variational', 'hidden_size': 128, 'num_lstm_layers': 2, 'kernel_size': 5, 'dropout': 0.2},
        {'model_type': 'variational', 'hidden_size': 128, 'num_lstm_layers': 2, 'kernel_size': 7, 'dropout': 0.2},
        
        # Variações de dropout
        {'model_type': 'variational', 'hidden_size': 128, 'num_lstm_layers': 2, 'kernel_size': 3, 'dropout': 0.1},
        {'model_type': 'variational', 'hidden_size': 128, 'num_lstm_layers': 2, 'kernel_size': 3, 'dropout': 0.3}
    ]
    
    # Resultados
    results = []
    
    # Executar experimentos
    for i, config in enumerate(configs):
        print(f"\nExperimento {i+1}/{len(configs)}")
        print(f"Configuração: {config}")
        
        # Treinar modelo
        model, history = train_neural_model(
            X_train_processed, y_train, 
            X_test_processed, y_test,
            model_type=config['model_type'],
            hidden_size=config['hidden_size'],
            num_lstm_layers=config['num_lstm_layers'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
            batch_size=64,
            epochs=100,  # Reduzido para experimentos mais rápidos
            patience=5,
            lr=0.001
        )
        
        # Avaliar no conjunto de validação
        device = next(model.parameters()).device
        
        # Preparar dados de validação
        X_val_processed = scaler.transform(X_val[:, :-n_derived_features])
        X_val_processed = np.hstack((X_val_processed, X_val[:, -n_derived_features:]))
        
        # Gerar realizações
        if config['model_type'] == 'deterministic':
            X_tensor = torch.FloatTensor(X_val_processed).to(device)
            with torch.no_grad():
                base_predictions = model(X_tensor).cpu().numpy()
                
            # Gerar realizações simples
            realizations = np.zeros((X_val.shape[0], NUM_REALIZATIONS, y_val.shape[1]))
            realizations[:, 0, :] = base_predictions
            
            for r in range(1, NUM_REALIZATIONS):
                noise_scale = 0.1 * r / NUM_REALIZATIONS
                realizations[:, r, :] = base_predictions + np.random.normal(0, noise_scale, base_predictions.shape)
                
        else:  # variational
            X_tensor = torch.FloatTensor(X_val_processed).to(device)
            with torch.no_grad():
                # Primeira realização (média)
                mu, _ = model.encode(X_tensor)
                base_predictions = mu.cpu().numpy()
                
                # Gerar realizações
                realizations = np.zeros((X_val.shape[0], NUM_REALIZATIONS, y_val.shape[1]))
                realizations[:, 0, :] = base_predictions
                
                # Gerar mais realizações
                std = torch.exp(0.5 * model.encode(X_tensor)[1])
                
                for r in range(1, NUM_REALIZATIONS):
                    eps = torch.randn_like(std)
                    sample = mu + eps * std
                    realizations[:, r, :] = sample.cpu().numpy()
        
        # Calcular NLL
        nll = calculate_nll_loss(y_val, realizations)
        
        # Armazenar resultados
        result = {
            'config': config,
            'val_loss': history['best_val_loss'],
            'best_epoch': history['best_epoch'],
            'nll': nll
        }
        results.append(result)
        
        print(f"NLL: {nll:.2f}")
    
    # Criar DataFrame com resultados
    df_results = pd.DataFrame(results)
    
    # Salvar resultados
    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/hyperparameter_search.csv', index=False)
    
    # Visualizar resultados
    plt.figure(figsize=(12, 8))
    
    # Plot NLL para cada configuração
    plt.subplot(2, 1, 1)
    indices = np.arange(len(results))
    plt.bar(indices, [r['nll'] for r in results])
    plt.axhline(y=-69, color='r', linestyle='--', label='Target NLL')
    plt.xticks(indices, [f"Config {i+1}" for i in indices])
    plt.ylabel('NLL')
    plt.title('NLL por Configuração')
    plt.legend()
    
    # Plot val_loss para cada configuração
    plt.subplot(2, 1, 2)
    plt.bar(indices, [r['val_loss'] for r in results])
    plt.xticks(indices, [f"Config {i+1}" for i in indices])
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss por Configuração')
    
    plt.tight_layout()
    plt.savefig('results/hyperparameter_search.png')
    plt.close()
    
    # Imprimir melhores configurações
    print("\nMelhores configurações por NLL (mais próximo de -69):")
    df_results['nll_diff'] = abs(df_results['nll'] - (-69))
    best_configs = df_results.sort_values('nll_diff').head(3)
    
    for i, row in best_configs.iterrows():
        print(f"Configuração {i+1}:")
        for k, v in row['config'].items():
            print(f"  {k}: {v}")
        print(f"  NLL: {row['nll']:.2f}")
        print(f"  Validation Loss: {row['val_loss']:.4f}")
        print()

def experiment_model_comparison():
    """
    Experimento para comparar modelo neural com ensemble tradicional
    """
    print("Experimento: Comparação entre modelo neural e ensemble tradicional")
    
    # Carregar e preparar dados
    train_df, _, input_cols, output_cols = load_data(TRAIN_PATH)
    X, X_val, y, y_val, n_derived_features = prepare_data(train_df, input_cols, output_cols)
    
    # Carregar modelo neural pré-treinado (assumindo que já foi treinado)
    if os.path.exists('neural_models/ensemble_info.pkl'):
        print("Carregando modelo neural pré-treinado...")
        
        # Preparar dados para modelo neural
        X_val_processed, scaler = prepare_data_for_neural_model(X_val, n_derived_features)
        
        # Gerar realizações com modelo neural
        neural_realizations = predict_with_neural_ensemble(
            X_val, n_derived_features, load_path='neural_models'
        )
        
        # Carregar realizações de ensemble tradicional (se disponíveis)
        if os.path.exists('results/traditional_ensemble_realizations.npz'):
            print("Carregando realizações do ensemble tradicional...")
            traditional_data = np.load('results/traditional_ensemble_realizations.npz')
            traditional_realizations = traditional_data['realizations']
            
            # Avaliar NLL
            neural_nll = calculate_nll_loss(y_val, neural_realizations)
            traditional_nll = calculate_nll_loss(y_val, traditional_realizations)
            
            print(f"NLL do Modelo Neural: {neural_nll:.2f}")
            print(f"NLL do Ensemble Tradicional: {traditional_nll:.2f}")
            
            # Testar diferentes pesos de combinação
            weights = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
            nll_values = []
            
            for w in weights:
                combined_realizations = combine_neural_and_ensemble_predictions(
                    neural_realizations, traditional_realizations, neural_weight=w
                )
                combined_nll = calculate_nll_loss(y_val, combined_realizations)
                nll_values.append(combined_nll)
                print(f"Peso Neural: {w:.1f}, NLL Combinado: {combined_nll:.2f}")
            
            # Encontrar melhor peso
            best_idx = np.argmin(np.abs(np.array(nll_values) - (-69)))
            best_weight = weights[best_idx]
            best_nll = nll_values[best_idx]
            
            print(f"Melhor peso neural: {best_weight:.2f}, NLL: {best_nll:.2f}")
            
            # Visualizar resultados
            plt.figure(figsize=(12, 6))
            plt.plot(weights, nll_values, 'o-', label='NLL Combinado')
            plt.axhline(y=-69, color='r', linestyle='--', label='Target NLL')
            plt.axhline(y=neural_nll, color='g', linestyle='--', label=f'Neural NLL ({neural_nll:.2f})')
            plt.axhline(y=traditional_nll, color='b', linestyle='--', label=f'Traditional NLL ({traditional_nll:.2f})')
            plt.axvline(x=best_weight, color='k', linestyle='--', label=f'Best Weight ({best_weight:.2f})')
            plt.xlabel('Neural Weight')
            plt.ylabel('NLL')
            plt.title('NLL por Peso do Modelo Neural na Combinação')
            plt.legend()
            plt.grid(True)
            plt.savefig('results/combination_weights.png')
            plt.close()
            
            # Visualizar exemplos de previsões
            visualize_prediction_examples(
                X_val, y_val, neural_realizations, traditional_realizations, n_derived_features
            )
            
        else:
            print("Arquivo de realizações do ensemble tradicional não encontrado.")
    else:
        print("Modelo neural pré-treinado não encontrado. Treine um modelo primeiro.")

def visualize_prediction_examples(X, y, neural_realizations, traditional_realizations, n_derived_features, num_examples=3):
    """
    Visualiza exemplos de previsões do modelo neural e ensemble tradicional
    
    Args:
        X: Features
        y: Valores verdadeiros
        neural_realizations: Realizações do modelo neural
        traditional_realizations: Realizações do ensemble tradicional
        n_derived_features: Número de características derivadas
        num_examples: Número de exemplos a visualizar
    """
    # Usar apenas características originais
    X_original = X[:, :-n_derived_features]
    
    # Escolher alguns exemplos aleatórios
    indices = np.random.choice(X_original.shape[0], num_examples, replace=False)
    
    plt.figure(figsize=(15, 5*num_examples))
    
    for i, idx in enumerate(indices):
        plt.subplot(num_examples, 1, i+1)
        
        # Dados conhecidos (entrada)
        plt.plot(range(-49, 1), X_original[idx], 'b-', label='Dados Conhecidos')
        
        # Valor verdadeiro (alvo)
        plt.plot(range(1, 301), y[idx], 'g-', label='Alvo Real')
        
        # Realizações do modelo neural
        for r in range(neural_realizations.shape[1]):
            if r == 0:
                plt.plot(range(1, 301), neural_realizations[idx, r, :], 'r-', alpha=0.7, label='Realização Base Neural')
            else:
                plt.plot(range(1, 301), neural_realizations[idx, r, :], 'r-', alpha=0.2)
        
        # Realizações do ensemble tradicional
        for r in range(traditional_realizations.shape[1]):
            if r == 0:
                plt.plot(range(1, 301), traditional_realizations[idx, r, :], 'c-', alpha=0.7, label='Realização Base Tradicional')
            else:
                plt.plot(range(1, 301), traditional_realizations[idx, r, :], 'c-', alpha=0.2)
        
        plt.axvline(x=0, color='k', linestyle='--')
        plt.title(f'Exemplo #{idx}')
        plt.xlabel('Posição X')
        plt.ylabel('Coordenada Z')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/prediction_comparison.png')
    plt.close()
    
    # Visualizar detalhe inicial (primeiras 60 posições, mais importantes para NLL)
    plt.figure(figsize=(15, 5*num_examples))
    
    for i, idx in enumerate(indices):
        plt.subplot(num_examples, 1, i+1)
        
        # Dados conhecidos (entrada)
        plt.plot(range(-49, 1), X_original[idx], 'b-', label='Dados Conhecidos')
        
        # Valor verdadeiro (alvo)
        plt.plot(range(1, 61), y[idx, :60], 'g-', label='Alvo Real')
        
        # Realizações do modelo neural
        for r in range(neural_realizations.shape[1]):
            if r == 0:
                plt.plot(range(1, 61), neural_realizations[idx, r, :60], 'r-', alpha=0.7, label='Realização Base Neural')
            else:
                plt.plot(range(1, 61), neural_realizations[idx, r, :60], 'r-', alpha=0.2)
        
        # Realizações do ensemble tradicional
        for r in range(traditional_realizations.shape[1]):
            if r == 0:
                plt.plot(range(1, 61), traditional_realizations[idx, r, :60], 'c-', alpha=0.7, label='Realização Base Tradicional')
            else:
                plt.plot(range(1, 61), traditional_realizations[idx, r, :60], 'c-', alpha=0.2)
        
        plt.axvline(x=0, color='k', linestyle='--')
        plt.title(f'Detalhe Inicial - Exemplo #{idx}')
        plt.xlabel('Posição X')
        plt.ylabel('Coordenada Z')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/prediction_comparison_detail.png')
    plt.close()

def experiment_latent_space_visualization():
    """
    Experimento para visualizar o espaço latente do modelo variacional
    """
    print("Experimento: Visualização do espaço latente do modelo variacional")
    
    # Verificar se o modelo variacional está disponível
    if not os.path.exists('neural_models/ensemble_info.pkl'):
        print("Modelo neural pré-treinado não encontrado. Treine um modelo primeiro.")
        return
    
    # Carregar informações do ensemble
    ensemble_info = joblib.load('neural_models/ensemble_info.pkl')
    if ensemble_info['model_type'] != 'variational':
        print("Este experimento requer um modelo variacional.")
        return
    
    # Carregar e preparar dados
    train_df, _, input_cols, output_cols = load_data(TRAIN_PATH)
    X, X_val, y, y_val, n_derived_features = prepare_data(train_df, input_cols, output_cols)
    
    # Preparar dados para modelo neural
    X_val_processed, scaler = prepare_data_for_neural_model(X_val, n_derived_features)
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar modelo
    checkpoint = torch.load('neural_models/neural_model_0.pt', map_location=device)
    config = checkpoint['config']
    
    # Criar modelo
    input_size = X_val.shape[1]
    output_size = 300  # Tamanho padrão de saída
    
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
    model.eval()
    
    # Converter dados para tensor
    X_tensor = torch.FloatTensor(X_val_processed).to(device)
    
    # Extrair vetores latentes (mu e logvar)
    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        std = torch.exp(0.5 * logvar)
        
        # Converter para numpy
        mu_np = mu.cpu().numpy()
        std_np = std.cpu().numpy()
    
    # Selecionar algumas posições representativas para visualização
    key_positions = [0, 50, 100, 150, 200, 250, 299]  # 0-indexed
    
    # Visualizar distribuição para cada posição-chave
    plt.figure(figsize=(15, 10))
    
    for i, pos in enumerate(key_positions):
        plt.subplot(len(key_positions), 1, i+1)
        
        # Histograma de médias para esta posição
        plt.hist(mu_np[:, pos], bins=30, alpha=0.7, label=f'Médias (μ)')
        
        # Marcar valor verdadeiro médio
        mean_true = np.mean(y_val[:, pos])
        plt.axvline(x=mean_true, color='r', linestyle='--', label=f'Média Real: {mean_true:.2f}')
        
        # Valor médio predito
        mean_pred = np.mean(mu_np[:, pos])
        plt.axvline(x=mean_pred, color='g', linestyle='--', label=f'Média Predita: {mean_pred:.2f}')
        
        # Desvio padrão médio
        mean_std = np.mean(std_np[:, pos])
        plt.title(f'Distribuição para Posição {pos+1} (σ média: {mean_std:.2f})')
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/latent_space_distribution.png')
    plt.close()
    
    # Visualizar a evolução do desvio padrão ao longo das posições
    # Isso indica como a incerteza aumenta com a distância
    mean_std_by_position = np.mean(std_np, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 301), mean_std_by_position)
    plt.title('Incerteza Média (σ) por Posição')
    plt.xlabel('Posição')
    plt.ylabel('Desvio Padrão Médio')
    plt.grid(True)
    plt.savefig('results/uncertainty_by_position.png')
    plt.close()
    
    # Visualizar amostras aleatórias do espaço latente
    num_samples = 5
    random_indices = np.random.choice(X_val.shape[0], num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(random_indices):
        plt.subplot(num_samples, 1, i+1)
        
        # Dados de entrada
        X_original = X_val[idx, :-n_derived_features]
        plt.plot(range(-49, 1), X_original, 'b-', label='Dados Conhecidos')
        
        # Valor verdadeiro
        plt.plot(range(1, 301), y_val[idx], 'g-', label='Alvo Real')
        
        # Obter parâmetros da distribuição
        x_single = X_tensor[idx:idx+1]
        with torch.no_grad():
            mu_i, logvar_i = model.encode(x_single)
            std_i = torch.exp(0.5 * logvar_i)
            
            # Gerar múltiplas amostras
            num_realizations = 10
            for r in range(num_realizations):
                eps = torch.randn_like(std_i)
                sample = mu_i + eps * std_i
                
                if r == 0:
                    plt.plot(range(1, 301), sample[0].cpu().numpy(), 'r-', label=f'Amostra {r+1}')
                else:
                    plt.plot(range(1, 301), sample[0].cpu().numpy(), 'r-', alpha=0.3)
        
        # Intervalos de confiança
        mean = mu_i[0].cpu().numpy()
        std = std_i[0].cpu().numpy()
        
        plt.fill_between(
            range(1, 301),
            mean - 2*std,
            mean + 2*std,
            color='r',
            alpha=0.2,
            label='Intervalo 95%'
        )
        
        plt.axvline(x=0, color='k', linestyle='--')
        plt.title(f'Amostra #{idx} - Múltiplas Realizações e Intervalo de Confiança')
        plt.xlabel('Posição')
        plt.ylabel('Valor')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/latent_space_samples.png')
    plt.close()

def main():
    """
    Função principal para executar os experimentos
    """
    print("Experimentos com modelos neurais para previsão geológica")
    
    # Criar diretório para resultados
    os.makedirs('results', exist_ok=True)
    
    # Menu de experimentos
    print("\nEscolha um experimento:")
    print("1. Busca de Hiperparâmetros")
    print("2. Comparação com Ensemble Tradicional")
    print("3. Visualização do Espaço Latente")
    
    choice = input("\nDigite o número do experimento (1-3): ")
    
    if choice == '1':
        experiment_hyperparameter_search()
    elif choice == '2':
        experiment_model_comparison()
    elif choice == '3':
        experiment_latent_space_visualization()
    else:
        print("Opção inválida.")

if __name__ == "__main__":
    main()
