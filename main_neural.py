import time
import warnings
import os
import numpy as np
import argparse
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_PATH, NUM_REALIZATIONS, TARGET_NLL
from utils.data_utils import load_data, prepare_data, prepare_test_data, prepare_submission
from utils.evaluation import analyze_variance_structure, calculate_nll_loss
from models.ensemble import train_advanced_ensemble_models, predict_with_advanced_ensemble
from enhanced_generator import generate_enhanced_realizations
from enhanced_calibration import optimize_region_specific_calibration, apply_calibrated_scaling
from visualization.plots import visualize_examples
from neural_integration import (
    train_neural_ensemble, 
    predict_with_neural_ensemble, 
    combine_neural_and_ensemble_predictions,
    integrate_neural_model_to_pipeline,
    prepare_data_for_neural_model
)

# Suprimir avisos
warnings.filterwarnings('ignore')

def main():
    """
    Função principal para executar o pipeline com integração de modelos neurais
    """
    # Parse argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Pipeline de previsão geológica com modelos neurais')
    parser.add_argument('--mode', type=str, default='integrated', choices=['integrated', 'neural_only', 'traditional'],
                      help='Modo de execução: integrated (neural+tradicional), neural_only, traditional')
    parser.add_argument('--train_new', action='store_true', default=True,
                      help='Treinar novos modelos neurais (default: True)')
    parser.add_argument('--model_type', type=str, default='variational', choices=['deterministic', 'variational'],
                      help='Tipo de modelo neural: deterministic ou variational')
    parser.add_argument('--ensemble_size', type=int, default=3,
                      help='Número de modelos no ensemble neural')
    parser.add_argument('--neural_weight', type=float, default=0.2,
                      help='Peso do modelo neural na combinação com ensemble tradicional (0-1)')
    
    args = parser.parse_args()
    
    # Garantir que avisos estejam desativados
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Iniciar timer para medir tempo total de execução
    start_time = time.time()
    
    # Criar diretório para modelos neurais se não existir
    if not os.path.exists('neural_models'):
        os.makedirs('neural_models')
    
    # Carregar dados
    print("\n1. Carregando dados...")
    train_df, _, input_cols, output_cols, test_df, test_ids = load_data(TRAIN_PATH, TEST_PATH)
    
    # Preparar dados com características derivadas
    print("\n2. Preparando dados...")
    X_train, X_val, y_train, y_val, n_derived_features = prepare_data(train_df, input_cols, output_cols)
    
    # Analisar estatísticas dos dados
    print("\n3. Analisando estrutura de variância...")
    variance_params = analyze_variance_structure(X_train, y_train, n_derived_features)
    
    #=============== PIPELINE NEURAL APENAS ===============#
    if args.mode == 'neural_only':
        print("\n4. Executando pipeline apenas com modelos neurais...")
        
        # Treinar modelos neurais
        if args.train_new:
            print("\n4.1. Treinando ensemble neural...")
            # Converter tipos numpy para Python nativos
            neural_models, neural_scaler, ensemble_info = train_neural_ensemble(
                X_train, y_train, X_val, y_val, n_derived_features,
                model_type=args.model_type,
                ensemble_size=args.ensemble_size,
                save_path='neural_models'
            )
        else:
            print("\n4.1. Carregando modelos neurais pré-treinados...")
            neural_models, neural_scaler, ensemble_info = None, None, None
        
        # Gerar previsões para validação usando modelos neurais
        print("\n4.2. Gerando previsões neurais para validação...")
        neural_realizations = predict_with_neural_ensemble(
            X_val, n_derived_features, models=neural_models,
            ensemble_info=ensemble_info, scaler=neural_scaler,
            num_realizations=NUM_REALIZATIONS, load_path='neural_models'
        )
        
        # Calcular NLL inicial
        print("\n4.3. Calculando NLL inicial do modelo neural...")
        neural_nll = calculate_nll_loss(y_val, neural_realizations)
        print(f"NLL do modelo neural: {neural_nll:.2f}")
        
        # Otimizar calibração
        print("\n4.4. Otimizando calibração específica por região...")
        calibration_params, calibrated_nll = optimize_region_specific_calibration(
            neural_realizations, y_val, target_nll=TARGET_NLL, max_iterations=20
        )
        print(f"NLL após calibração: {calibrated_nll:.2f}")
        
        # Visualizar exemplos
        print("\n4.5. Visualizando exemplos...")
        visualize_examples(X_val, y_val, neural_realizations, n_derived_features)
        
        # Preparar dados de teste
        print("\n5. Preparando dados de teste...")
        X_test_enhanced = prepare_test_data(test_df, input_cols, n_derived_features)
        
        # Gerar previsões para teste
        print("\n6. Gerando previsões para teste...")
        test_neural_realizations = predict_with_neural_ensemble(
            X_test_enhanced, n_derived_features, models=neural_models,
            ensemble_info=ensemble_info, scaler=neural_scaler,
            num_realizations=NUM_REALIZATIONS, load_path='neural_models'
        )
        
        # Aplicar calibração otimizada
        print("\n7. Aplicando calibração...")
        test_calibrated_realizations = apply_calibrated_scaling(
            test_neural_realizations, test_neural_realizations[:, 0, :], calibration_params
        )
        
        # Preparar e salvar arquivo de submissão
        print("\n8. Preparando arquivo de submissão...")
        submission_path = SUBMISSION_PATH.replace(".csv", "_neural_only.csv")
        prepare_submission(test_ids, test_calibrated_realizations, submission_path)
    
    #=============== PIPELINE INTEGRADO ===============#
    elif args.mode == 'integrated':
        print("\n4. Integrando modelo neural ao pipeline...")
        
        # Opção 1: Treinar ensemble tradicional primeiro
        print("\n4.1. Treinando ensemble tradicional...")
        models, scaler, key_positions, n_derived_features = train_advanced_ensemble_models(
            X_train, y_train, n_derived_features
        )
        
        # Gerar previsões base para validação usando ensemble avançado
        print("\n4.2. Gerando previsões de validação com ensemble tradicional...")
        val_base_predictions = predict_with_advanced_ensemble(
            X_val, models, scaler, key_positions, n_derived_features
        )
        
        # Gerar realizações para ensemble tradicional
        print("\n4.3. Gerando realizações com ensemble tradicional...")
        traditional_realizations = generate_enhanced_realizations(
            X_val, val_base_predictions, variance_params, n_derived_features, NUM_REALIZATIONS
        )
        
        # Calcular NLL inicial com ensemble tradicional
        print("\n4.4. Calculando NLL inicial do ensemble tradicional...")
        traditional_nll = calculate_nll_loss(y_val, traditional_realizations)
        print(f"NLL do ensemble tradicional: {traditional_nll:.2f}")
        
        # Treinar explicitamente o ensemble neural se necessário
        if args.train_new:
            print("\n4.5. Treinando ensemble de modelos neurais...")
            neural_models, neural_scaler, ensemble_info = train_neural_ensemble(
                X_train, y_train, X_val, y_val, n_derived_features,
                model_type=args.model_type,
                ensemble_size=args.ensemble_size,
                save_path='neural_models'
            )
        else:
            print("\n4.5. Carregando modelos neurais pré-treinados...")
            neural_models, neural_scaler, ensemble_info = None, None, None
        
        # Gerar realizações neurais
        print("\n4.6. Gerando realizações com modelo neural...")
        neural_realizations = predict_with_neural_ensemble(
            X_val, n_derived_features, models=neural_models,
            ensemble_info=ensemble_info, scaler=neural_scaler,
            num_realizations=NUM_REALIZATIONS, load_path='neural_models'
        )
        
        # Calcular NLL do modelo neural
        neural_nll = calculate_nll_loss(y_val, neural_realizations)
        print(f"NLL do modelo neural: {neural_nll:.2f}")
        
        # Combinar realizações
        print("\n4.7. Combinando realizações neurais e tradicionais...")
        combined_realizations = combine_neural_and_ensemble_predictions(
            neural_realizations, traditional_realizations, neural_weight=args.neural_weight
        )
        
        # Calibrar conjunto combinado
        print("\n4.8. Calibrando realizações combinadas...")
        calibration_params, calibrated_nll = optimize_region_specific_calibration(
            combined_realizations, y_val, target_nll=TARGET_NLL, max_iterations=20
        )
        print(f"NLL combinado após calibração: {calibrated_nll:.2f}")
        
        # Visualizar alguns exemplos
        print("\n5. Visualizando exemplos...")
        visualize_examples(X_val, y_val, combined_realizations, n_derived_features)
        
        # Preparar dados de teste
        print("\n6. Preparando dados de teste...")
        X_test_enhanced = prepare_test_data(test_df, input_cols, n_derived_features)
        
        # Gerar previsões para teste usando ensemble tradicional
        print("\n7. Gerando previsões de teste com ensemble tradicional...")
        test_base_predictions = predict_with_advanced_ensemble(
            X_test_enhanced, models, scaler, key_positions, n_derived_features
        )
        
        # Gerar previsões para teste usando modelo neural
        print("\n8. Gerando previsões de teste com modelo neural...")
        neural_test_realizations = predict_with_neural_ensemble(
            X_test_enhanced, n_derived_features, models=neural_models,
            ensemble_info=ensemble_info, scaler=neural_scaler,
            load_path='neural_models'
        )
        
        # Gerar realizações para teste com ensemble tradicional
        print("\n9. Gerando realizações de teste com ensemble tradicional...")
        traditional_test_realizations = generate_enhanced_realizations(
            X_test_enhanced, test_base_predictions, variance_params, 
            n_derived_features, NUM_REALIZATIONS
        )
        
        # Combinar realizações neurais e tradicionais
        print("\n10. Combinando realizações neurais e tradicionais para teste...")
        combined_test_realizations = combine_neural_and_ensemble_predictions(
            neural_test_realizations, traditional_test_realizations,
            neural_weight=args.neural_weight, num_realizations=NUM_REALIZATIONS
        )
        
        # Aplicar calibração final
        print("\n11. Aplicando calibração final...")
        final_test_realizations = apply_calibrated_scaling(
            combined_test_realizations, combined_test_realizations[:, 0, :], 
            calibration_params
        )
        
        # Preparar e salvar arquivo de submissão
        print("\n12. Preparando arquivo de submissão...")
        submission_path = SUBMISSION_PATH.replace(".csv", "_neural_integrated.csv")
        prepare_submission(test_ids, final_test_realizations, submission_path)
    
    #=============== PIPELINE TRADICIONAL ===============#
    else:  # args.mode == 'traditional'
        # Pipeline original
        print("\n4. Treinando modelos de ensemble avançado...")
        models, scaler, key_positions, n_derived_features = train_advanced_ensemble_models(
            X_train, y_train, n_derived_features
        )
        
        # Gerar previsões base para validação usando ensemble avançado
        print("\n5. Gerando previsões de validação...")
        val_base_predictions = predict_with_advanced_ensemble(
            X_val, models, scaler, key_positions, n_derived_features
        )
        
        # Gerar realizações aprimoradas com maior diversidade para validação
        print(f"\n6. Gerando realizações aprimoradas para validação (n={NUM_REALIZATIONS})...")
        val_realizations = generate_enhanced_realizations(
            X_val, val_base_predictions, variance_params, n_derived_features, NUM_REALIZATIONS
        )
        
        # Calcular NLL inicial
        print("\n7. Calculando NLL inicial...")
        nll_loss = calculate_nll_loss(y_val, val_realizations)
        print(f"Negative Log Likelihood (NLL) inicial: {nll_loss}")
        
        # Otimizar calibração específica por região para target NLL
        print("\n8. Otimizando calibração específica por região para target NLL...")
        calibration_params, best_nll = optimize_region_specific_calibration(
            val_realizations, y_val, target_nll=TARGET_NLL, max_iterations=20
        )
        
        # Visualizar alguns exemplos
        print("\n9. Visualizando exemplos...")
        visualize_examples(X_val, y_val, val_realizations, n_derived_features)
        
        # Preparar dados de teste
        print("\n10. Preparando dados de teste...")
        X_test_enhanced = prepare_test_data(test_df, input_cols, n_derived_features)
        
        # Gerar previsões base para teste usando ensemble avançado
        print("\n11. Gerando previsões para teste...")
        test_base_predictions = predict_with_advanced_ensemble(
            X_test_enhanced, models, scaler, key_positions, n_derived_features
        )
        
        # Gerar realizações aprimoradas para teste
        print(f"\n12. Gerando realizações aprimoradas para teste (n={NUM_REALIZATIONS})...")
        test_realizations = generate_enhanced_realizations(
            X_test_enhanced, test_base_predictions, variance_params, 
            n_derived_features, NUM_REALIZATIONS
        )
        
        # Aplicar calibração otimizada
        print("\n13. Aplicando calibração...")
        test_realizations = apply_calibrated_scaling(
            test_realizations, test_base_predictions, calibration_params
        )
        
        # Preparar e salvar arquivo de submissão
        print("\n14. Preparando arquivo de submissão...")
        submission_path = SUBMISSION_PATH.replace(".csv", "_traditional.csv")
        prepare_submission(test_ids, test_realizations, submission_path)
    
    # Calcular e exibir tempo de execução
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nProcesso concluído com sucesso!")
    print(f"Tempo total de execução: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Submissão salva em: {submission_path}")

if __name__ == "__main__":
    main()