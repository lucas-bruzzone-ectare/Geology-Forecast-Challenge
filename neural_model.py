import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from config import SEED

# Para reprodutibilidade
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class GeoSequenceDataset(TensorDataset):
    """
    Dataset personalizado para dados geológicos de sequência
    """
    def __init__(self, X, y=None):
        # Converter para tensores
        X_tensor = torch.FloatTensor(X)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y)
            super().__init__(X_tensor, y_tensor)
        else:
            super().__init__(X_tensor)

class GeoCNN_LSTM(nn.Module):
    """
    Modelo híbrido com CNN 1D seguido por LSTM para previsão de sequências geológicas
    """
    def __init__(self, input_size, output_size=300, hidden_size=128, 
                 num_lstm_layers=2, dropout=0.2, kernel_size=3):
        super(GeoCNN_LSTM, self).__init__()
        
        # Camada convolucional 1D para extrair características locais
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Camada de pooling para reduzir a dimensionalidade
        self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        
        # Camada LSTM para capturar dependências sequenciais
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Camadas fully connected para saída
        lstm_output_size = hidden_size * 2  # bidirectional
        self.fc1 = nn.Linear(lstm_output_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)
        
        # Ativações
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Reshape para conv1d (batch, channels, seq_len)
        x = x.view(batch_size, 1, seq_len)
        
        # Aplicar camadas convolucionais
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape para LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # Aplicar LSTM
        x, _ = self.lstm(x)
        
        # Usar apenas a saída do último passo de tempo
        x = x[:, -1, :]
        
        # Aplicar camadas fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class VariationalGeoCNN_LSTM(nn.Module):
    """
    Modelo variacional para geração de múltiplas realizações
    """
    def __init__(self, input_size, output_size=300, hidden_size=128, 
                 num_lstm_layers=2, dropout=0.2, kernel_size=3):
        super(VariationalGeoCNN_LSTM, self).__init__()
        
        # Camada convolucional 1D para extrair características locais
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Camada de pooling
        self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        
        # Camada LSTM
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Variational: duas saídas para média e log-variância
        lstm_output_size = hidden_size * 2  # bidirectional
        self.fc_mu = nn.Linear(lstm_output_size, output_size)
        self.fc_logvar = nn.Linear(lstm_output_size, output_size)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)
        
        # Ativações
        self.relu = nn.ReLU()
    
    def encode(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Reshape para conv1d (batch, channels, seq_len)
        x = x.view(batch_size, 1, seq_len)
        
        # Aplicar camadas convolucionais
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape para LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # Aplicar LSTM
        x, _ = self.lstm(x)
        
        # Usar apenas a saída do último passo de tempo
        x = x[:, -1, :]
        
        # Calcular média e log-variância
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Trick de reparametrização para permitir o backpropagation
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def sample(self, x, num_samples=1):
        """
        Gera múltiplas realizações
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            
            samples = []
            for _ in range(num_samples):
                eps = torch.randn_like(std)
                sample = mu + eps * std
                samples.append(sample)
                
            return torch.stack(samples, dim=1)  # [batch_size, num_samples, output_size]


def train_neural_model(X_train, y_train, X_val, y_val, model_type='deterministic', 
                       batch_size=64, epochs=100, lr=0.001, patience=10,
                       hidden_size=128, num_lstm_layers=2, dropout=0.2, 
                       kernel_size=3, device=None):
    """
    Treina o modelo neural
    
    Args:
        X_train: Features de treinamento
        y_train: Alvos de treinamento
        X_val: Features de validação
        y_val: Alvos de validação
        model_type: 'deterministic' ou 'variational'
        batch_size: Tamanho do batch
        epochs: Número de épocas
        lr: Taxa de aprendizado
        patience: Paciência para early stopping
        hidden_size: Tamanho do estado oculto LSTM
        num_lstm_layers: Número de camadas LSTM
        dropout: Taxa de dropout
        kernel_size: Tamanho do kernel para CNN
        device: Dispositivo (cuda/cpu)
        
    Returns:
        Modelo treinado e histórico de treinamento
    """
    # Determinar o dispositivo
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Treinando no dispositivo: {device}")
    
    # Criar datasets
    train_dataset = GeoSequenceDataset(X_train, y_train)
    val_dataset = GeoSequenceDataset(X_val, y_val)
    
    # Criar dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Inicializar modelo
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    if model_type == 'deterministic':
        model = GeoCNN_LSTM(
            input_size=input_size, 
            output_size=output_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            kernel_size=kernel_size
        ).to(device)
        criterion = nn.MSELoss()
    else:  # variational
        model = VariationalGeoCNN_LSTM(
            input_size=input_size, 
            output_size=output_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            kernel_size=kernel_size
        ).to(device)
        
        # Função de perda para modelo variacional (MSE + KL divergence)
        def vae_loss_fn(pred, target, mu, logvar):
            # Reconstrução
            mse_loss = nn.functional.mse_loss(pred, target, reduction='sum')
            
            # Regularização: KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Balancear os termos
            beta = 0.001  # Hiperparâmetro para ajustar a importância do termo KL
            return mse_loss + beta * kl_loss
        
        criterion = vae_loss_fn
    
    # Otimizador e scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Histórico de treinamento
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Early stopping
    counter = 0
    
    # Loop de treinamento
    for epoch in range(epochs):
        # Treino
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'deterministic':
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
            else:  # variational
                y_pred, mu, logvar = model(X_batch)
                loss = criterion(y_pred, y_batch, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validação
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                if model_type == 'deterministic':
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                else:  # variational
                    y_pred, mu, logvar = model(X_batch)
                    loss = criterion(y_pred, y_batch, mu, logvar)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Atualizar histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Atualizar scheduler
        scheduler.step(val_loss)
        
        # Verificar se é o melhor modelo
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            # Salvar o melhor modelo
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
        
        # Exibir progresso
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping em epoch {epoch+1}")
            break
    
    # Carregar o melhor modelo
    model.load_state_dict(best_model)
    
    # Visualizar curva de aprendizado
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label=f'Best epoch: {history["best_epoch"]+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curve')
    
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        model.eval()
        # Selecionar um exemplo aleatório
        idx = np.random.randint(0, len(X_val))
        X_sample = torch.FloatTensor(X_val[idx:idx+1]).to(device)
        y_true = y_val[idx]
        
        if model_type == 'deterministic':
            y_pred = model(X_sample).cpu().numpy()[0]
        else:
            # Gerar múltiplas realizações
            y_samples = model.sample(X_sample, num_samples=5).cpu().numpy()[0]
            y_pred = y_samples[0]  # Pegar primeira realização para visualização
    
    plt.plot(range(1, output_size+1), y_true, 'g-', label='True')
    plt.plot(range(1, output_size+1), y_pred, 'r-', label='Predicted')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Example Prediction')
    
    plt.tight_layout()
    plt.savefig('neural_model_training.png')
    
    return model, history


def predict_with_neural_model(model, X, model_type='deterministic', num_realizations=10, device=None):
    """
    Gera previsões com o modelo neural
    
    Args:
        model: Modelo treinado
        X: Features
        model_type: 'deterministic' ou 'variational'
        num_realizations: Número de realizações (para modelo variacional)
        device: Dispositivo (cuda/cpu)
        
    Returns:
        Array de previsões
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Converter para tensor
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Fazer previsões
    model.eval()
    with torch.no_grad():
        if model_type == 'deterministic':
            # Para modelo determinístico, usamos a mesma saída como realização base
            base_predictions = model(X_tensor).cpu().numpy()
            
            # Criar array para realizações (n_samples, num_realizations, n_outputs)
            realizations = np.zeros((X.shape[0], num_realizations, base_predictions.shape[1]))
            
            # Primeira realização é a previsão base
            realizations[:, 0, :] = base_predictions
            
            # Para outras realizações, adicionamos ruído (será melhorado com a função de geração)
            # Isso é apenas um placeholder para integração com o sistema atual
            for r in range(1, num_realizations):
                noise_scale = 0.1 * r / num_realizations
                realizations[:, r, :] = base_predictions + np.random.normal(0, noise_scale, base_predictions.shape)
        else:
            # Para modelo variacional, geramos múltiplas realizações diretamente
            realizations = model.sample(X_tensor, num_samples=num_realizations).cpu().numpy()
            # Transpor para formato (n_samples, num_realizations, n_outputs)
            realizations = np.transpose(realizations, (0, 1, 2))
    
    return realizations
