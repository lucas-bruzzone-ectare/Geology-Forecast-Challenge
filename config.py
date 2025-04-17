import numpy as np
import random
import warnings
import os

# Configurações globais
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Suprimir avisos do LightGBM
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['LIGHTGBM_VERBOSE'] = '-1'  # Diminui a verbosidade do LightGBM

# Número de realizações a prever
NUM_REALIZATIONS = 10

# Parâmetros para cálculo da métrica de avaliação
LOG_SLOPES = [1.0406028049510443, 0.0, 7.835345062351012]
LOG_OFFSETS = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]

# Configurações de arquivos
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMISSION_PATH = "submission_advanced_ensemble.csv"

# Configurações de visualização
VISUALIZATION_PATH_VARIANCE = "variance_analysis_refined.png"
VISUALIZATION_PATH_PREDICTIONS = "validation_predictions_corrected.png"
VISUALIZATION_PATH_CALIBRATION = "calibration_analysis.png"


# Configurações de hiperparâmetros
DEFAULT_XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': SEED
}

DEFAULT_LGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'random_state': SEED
}

DEFAULT_GBR_PARAMS = {
    'n_estimators': 150,
    'max_depth': 4,
    'learning_rate': 0.05,
    'loss': 'squared_error',
    'subsample': 0.8,
    'random_state': SEED
}

DEFAULT_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_split': 5,
    'random_state': SEED
}

DEFAULT_RIDGE_PARAMS = {
    'alpha': 1.0,
    'solver': 'auto',
    'random_state': SEED
}

DEFAULT_ELASTIC_PARAMS = {
    'alpha': 0.5,
    'l1_ratio': 0.5,
    'random_state': SEED,
    'max_iter': 1000
}

# Parâmetros para otimização bayesiana
XGB_PARAM_SPACE = {
    'n_estimators': (50, 200),
    'max_depth': (3, 6),
    'learning_rate': (0.01, 0.2),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'min_child_weight': (1, 5),
    'gamma': (0.0, 5.0),
    'reg_alpha': (0.0, 1.0),
    'reg_lambda': (0.0, 1.0)
}

LGB_PARAM_SPACE = {
    'n_estimators': (50, 200),
    'max_depth': (3, 6),
    'learning_rate': (0.01, 0.2),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'min_child_samples': (5, 30),
    'reg_alpha': (0.0, 1.0),
    'reg_lambda': (0.0, 1.0)
}

GBR_PARAM_SPACE = {
    'n_estimators': (50, 200),
    'max_depth': (3, 6),
    'learning_rate': (0.01, 0.2),
    'subsample': (0.6, 1.0)
}

RF_PARAM_SPACE = {
    'n_estimators': (50, 200),
    'max_depth': (3, 8),
    'min_samples_split': (2, 10)
}

# Estratégias para geração de realizações
REALIZATION_STRATEGIES = [
    # Name, base scale, correlation length, trend, trend amplitude
    ("high_fidelity", 0.8, None, None, 0.0),
    ("low_variance", 0.6, 1.2, None, 0.0),
    ("high_variance", 1.0, 0.6, None, 0.0),
    ("upward_trend", 0.8, 1.0, "upward", 1.5),
    ("downward_trend", 0.8, 1.0, "downward", 1.5),
    ("late_upward_trend", 0.8, 1.0, "late_upward", 2.0),
    ("late_downward_trend", 0.8, 1.0, "late_downward", 2.0),
    ("short_oscillation", 0.7, 0.5, "oscillatory", 1.0),
    ("long_oscillation", 0.7, 1.5, "oscillatory", 1.2),
]

# NLL alvo para calibração
TARGET_NLL = -69