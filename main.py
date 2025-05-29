# Importamos las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Configuración de semilla para reproducibilidad
torch.manual_seed(42)


# Función para crear los DataLoaders
def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32, val_split=0.1):
    """
    Crea DataLoaders de PyTorch para los conjuntos de entrenamiento, validación y prueba.
    Args:
        X_train (array-like): Características de entrenamiento.
        y_train (array-like): Etiquetas de entrenamiento.
        X_test (array-like): Características de prueba.
        y_test (array-like): Etiquetas de prueba.
        batch_size (int, opcional): Número de muestras por lote. Por defecto es 32.
        val_split (float, opcional): Fracción de los datos de entrenamiento para validación. Por defecto es 0.1.
    Returns:
        tuple: Una tupla que contiene:
            - train_loader (DataLoader): DataLoader para el conjunto de entrenamiento.
            - val_loader (DataLoader): DataLoader para el conjunto de validación.
            - test_loader (DataLoader): DataLoader para el conjunto de prueba.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convertir a tensores
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Split para validación
    val_size = int(val_split * len(X_train))
    train_size = len(X_train) - val_size
    
    X_train_split = X_train_tensor[:train_size]
    y_train_split = y_train_tensor[:train_size]
    X_val = X_train_tensor[train_size:]
    y_val = y_train_tensor[train_size:]
    
    # Crear datasets
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Función de entrenamiento
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10, model_save_path='models/best_model.pth'):
    """
    Entrena un modelo de PyTorch usando los DataLoaders de entrenamiento y validación, con soporte para early stopping y guardado del mejor modelo.
    Args:
        model (torch.nn.Module): El modelo de PyTorch a entrenar.
        train_loader (torch.utils.data.DataLoader): DataLoader para el conjunto de entrenamiento.
        val_loader (torch.utils.data.DataLoader): DataLoader para el conjunto de validación.
        criterion (torch.nn.Module): Función de pérdida a optimizar.
        optimizer (torch.optim.Optimizer): Optimizador para actualizar los parámetros del modelo.
        num_epochs (int, opcional): Número máximo de épocas de entrenamiento. Por defecto es 50.
        patience (int, opcional): Número de épocas a esperar sin mejora en la pérdida de validación antes de detener el entrenamiento. Por defecto es 10.
        model_save_path (str, opcional): Ruta de archivo para guardar los pesos del mejor modelo. Por defecto es 'models/best_model.pth'.
    Returns:
        tuple: Una tupla con dos listas:
            - train_losses (list of float): Pérdida de entrenamiento por época.
            - val_losses (list of float): Pérdida de validación por época.
    """
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Fase de entrenamiento
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        train_losses.append(train_loss)
        
        # Fase de validación
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                val_loss += loss.item()
                num_val_batches += 1
        
        val_loss /= num_val_batches
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Guardar mejor modelo
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load(model_save_path))
    
    return train_losses, val_losses

# Función de evaluación
def evaluate_model(model, test_loader, scaler, criterion):
    """
    Evalúa un modelo de PyTorch entrenado en un conjunto de prueba, calcula la pérdida promedio y retorna los valores reales y predichos (ambos desnormalizados).
    Args:
        model (torch.nn.Module): Modelo de PyTorch entrenado a evaluar.
        test_loader (torch.utils.data.DataLoader): DataLoader para el conjunto de prueba.
        scaler (sklearn.preprocessing.MinMaxScaler o similar): Escalador usado para normalizar los valores objetivo durante el entrenamiento, usado aquí para desnormalizar.
        criterion (torch.nn.Module): Función de pérdida utilizada para calcular la pérdida de prueba.
    Returns:
        tuple: (y_true, y_pred, test_loss)
            y_true (np.ndarray): Array de valores reales (desnormalizados).
            y_pred (np.ndarray): Array de valores predichos (desnormalizados).
            test_loss (float): Pérdida promedio sobre el conjunto de prueba.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            test_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    test_loss /= num_batches
    
    # Convertir a arrays y desnormalizar
    y_pred = np.array(all_predictions).reshape(-1, 1)
    y_true = np.array(all_targets).reshape(-1, 1)
    
    y_pred = scaler.inverse_transform(y_pred)
    y_true = scaler.inverse_transform(y_true)
    
    return y_true, y_pred, test_loss

# Código principal
def main_training_pipeline(X_train, y_train, X_test, y_test, model_save_path):
    """
    Entrena, valida y evalúa un modelo LSTM para la predicción del precio de Bitcoin.
    Esta función orquesta todo el pipeline de entrenamiento: crea los DataLoaders, inicializa el modelo,
    entrena con early stopping, evalúa en el conjunto de prueba y calcula métricas de regresión.
    Args:
        X_train (np.ndarray o torch.Tensor): Datos de características de entrenamiento.
        y_train (np.ndarray o torch.Tensor): Datos objetivo de entrenamiento.
        X_test (np.ndarray o torch.Tensor): Datos de características de prueba.
        y_test (np.ndarray o torch.Tensor): Datos objetivo de prueba.
        model_save_path (str): Ruta para guardar el checkpoint del mejor modelo.
    Returns:
        model (torch.nn.Module): El modelo LSTM entrenado.
        train_losses (list of float): Valores de pérdida de entrenamiento por época.
        val_losses (list of float): Valores de pérdida de validación por época.
        y_true (np.ndarray): Valores reales del conjunto de prueba.
        y_pred (np.ndarray): Valores predichos del conjunto de prueba.
    """
    # Crear DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=32, val_split=0.1
    )
    
    # Inicializar modelo
    model = BitcoinLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenar modelo
    print("Iniciando entrenamiento...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=50, patience=10, model_save_path=model_save_path
        )
    
    # Evaluar modelo
    y_true, y_pred, test_loss = evaluate_model(model, test_loader, scaler, criterion)
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    
    return model, train_losses, val_losses, y_true, y_pred

# Definición del modelo LSTM para Bitcoin
class BitcoinLSTM(nn.Module):
    """
    BitcoinLSTM es un módulo de red neuronal en PyTorch diseñado para predicción de series temporales usando una arquitectura LSTM de dos capas,
    seguido de capas totalmente conectadas. Es adecuado para tareas como la predicción del precio de Bitcoin.
    Parámetros:
        input_size (int): Número de características de entrada por paso temporal. Por defecto es 1.
        hidden_size1 (int): Número de unidades ocultas en la primera capa LSTM. Por defecto es 100.
        hidden_size2 (int): Número de unidades ocultas en la segunda capa LSTM. Por defecto es 50.
        num_layers1 (int): Número de capas apiladas en el primer bloque LSTM. Por defecto es 1.
        num_layers2 (int): Número de capas apiladas en el segundo bloque LSTM. Por defecto es 1.
        output_size (int): Número de características de salida. Por defecto es 1.
        dropout_rate (float): Tasa de dropout aplicada después de cada capa LSTM. Por defecto es 0.2.
    Entrada del método forward:
        x (torch.Tensor): Tensor de entrada de forma (batch_size, sequence_length, input_size).
    Salida del método forward:
        torch.Tensor: Tensor de salida de forma (batch_size, output_size), representando la predicción para cada secuencia de entrada.
    Arquitectura:
        - Dos capas LSTM con dropout opcional y número configurable de capas y unidades ocultas.
        - Dropout aplicado después de cada capa LSTM para evitar sobreajuste.
        - La salida del último paso temporal de la segunda LSTM pasa por dos capas totalmente conectadas.
        - Se aplica activación ReLU entre las capas totalmente conectadas.
    Ejemplo:
        model = BitcoinLSTM(input_size=1, hidden_size1=100, hidden_size2=50)
        output = model(torch.randn(32, 10, 1))  # batch_size=32, sequence_length=10
    """
    def __init__(self, input_size=1, hidden_size1=100, hidden_size2=50, num_layers1=1, num_layers2=1, output_size=1, dropout_rate=0.2):
        super(BitcoinLSTM, self).__init__()
        
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        
        # Primera capa LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers1, 
                            batch_first=True, dropout=dropout_rate if num_layers1 > 1 else 0)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Segunda capa LSTM
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers2, 
                            batch_first=True, dropout=dropout_rate if num_layers2 > 1 else 0)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Capas densas para la salida
        self.fc1 = nn.Linear(hidden_size2, 25)
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        # Primera capa LSTM
        h0_1 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(x.device)
        c0_1 = torch.zeros(self.num_layers1, x.size(0), self.hidden_size1).to(x.device)
        
        out, _ = self.lstm1(x, (h0_1, c0_1))
        out = self.dropout1(out)
        
        # Segunda capa LSTM
        h0_2 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(x.device)
        c0_2 = torch.zeros(self.num_layers2, x.size(0), self.hidden_size2).to(x.device)
        
        out, _ = self.lstm2(out, (h0_2, c0_2))
        out = self.dropout2(out)
        
        # Tome solo la última salida
        out = out[:, -1, :]
        
        # Capas densas para la salida
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

# Variables de configuración
models_dir = './models'
data_dir = './data'
nombre_modelo = 'bitcoin_lstm_model.pth'
model_save_path = f"{models_dir}/{nombre_modelo}"
results_dir = './results'


# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Cargamos el dataset de Bitcoin desde un archivo CSV (asumiendo que ya lo has descargado de Kaggle)
data = pd.read_csv(data_dir+'/'+'btcusd_1-min_data.csv')  # Adjust filename as per your Kaggle download
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
data.set_index('Timestamp', inplace=True)

data.head()

daily_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].resample('D').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna() # Resampleo los datos a diario y eliminar filas con NaN

daily_data.head()

# Normalizar los datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_data['Close'].values.reshape(-1, 1))

# Crear secuencias para LSTM (retrospectiva de 60 días)
seq_length = 60
def create_sequences(data, seq_length):
    """
    Genera secuencias de entrada-salida a partir de una serie temporal para aprendizaje supervisado.

    Args:
        data (array-like): Serie temporal a dividir en secuencias.
        seq_length (int): Longitud de cada secuencia de entrada.

    Returns:
        tuple: Una tupla (X, y) donde:
            - X (np.ndarray): Array de secuencias de entrada de forma (num_secuencias, seq_length).
            - y (np.ndarray): Array de valores objetivo correspondientes a cada secuencia de entrada.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length)

# Dividido en conjuntos de entrenamiento y prueba (división 80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
#print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Ejecutar el pipeline
model, train_losses, val_losses, y_true, y_pred = main_training_pipeline(X_train, y_train, X_test, y_test, model_save_path)

# Visualizar las pérdidas de entrenamiento y validación
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Pérdida de entrenamiento')
plt.plot(val_losses, label='Pérdida de validación')
plt.title('Pérdidas de entrenamiento y validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
# Guardo la figura de pérdidas
plt.savefig(f"{results_dir}/grafico_perdida.png")
plt.show()

# Grafico valores reales vs predicciones
plt.figure(figsize=(14, 8))
plt.plot(daily_data.index[-len(y_test):], scaler.inverse_transform(y_true), label='Precios Reales', color='#1E90FF', linewidth=2)
plt.plot(daily_data.index[-len(y_test):], scaler.inverse_transform(y_pred), label='Precios Predichos', color='#FF4500', linewidth=2)
plt.title('Predicción de precios de Bitcoin - Real vs Predicho', fontsize=16, pad=15)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Precio (USD)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
# Guardo la figura de predicciones
plt.savefig(f"{results_dir}/predicciones_plot.png")
plt.show()


# Predicciones para los próximos 5 días
model.eval()
with torch.no_grad():
    # Obtener la última secuencia de los datos de entrenamiento
    last_sequence = torch.FloatTensor(scaled_data[-seq_length:]).reshape(1, seq_length, 1).to(device)
    
    # Generar predicciones para los próximos 5 días
    future_predictions = []
    n_future = 5
    
    current_sequence = last_sequence.clone()
    for _ in range(n_future):
        next_pred = model(current_sequence)
        future_predictions.append(next_pred.cpu().item())
        
        # Actualizar la secuencia eliminando el primer elemento y añadiendo predicción
        # next_pred shape: (1, 1), necesitamos cambiar su forma a (1, 1, 1)
        next_pred_reshaped = next_pred.unsqueeze(2)  # Shape: (1, 1, 1)
        current_sequence = torch.cat([current_sequence[:, 1:, :], next_pred_reshaped], dim=1)

# Transformar inversamente las predicciones
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Crea un DataFrame con fechas y precios previstos
last_date = daily_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future, freq='D')
future_df = pd.DataFrame({
    'Fecha': future_dates,
    'Precio_de_cierre_previsto_en_USD': future_predictions.flatten()
})

# Mostrar el DataFrame
print("\nPrecios de cierre previstos de Bitcoin para los próximos 5 días:")
print(future_df.to_string(index=False))
# Guardo el DataFrame de predicciones futuras
future_df.to_csv(f"{results_dir}/predicciones_futuras.csv", index=False)

# Graficar predicciones futuras
plt.figure(figsize=(14, 8))
# Graficar los últimos 100 días de datos reales
recent_data = daily_data['Close'][-100:]
plt.plot(recent_data.index, recent_data.values, label='Precios Históricos', color='#1E90FF', linewidth=2)

# Graficar predicciones futuras
plt.plot(future_dates, future_predictions.flatten(), label='Predicciones Futuras', 
         color='#FF4500', linewidth=2, marker='o', markersize=6)
plt.title('Predicción del precio de Bitcoin: histórico y futuro', fontsize=16, pad=15)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Precio (USD)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=last_date, color='red', linestyle='--', alpha=0.7, label='Inicio de la predicción')
# Guardar la figura de predicciones futuras
plt.savefig(f"{results_dir}/grafico_predicciones_futuras.png")
plt.show()
