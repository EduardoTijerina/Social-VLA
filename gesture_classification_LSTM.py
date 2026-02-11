#!/usr/bin/env python3
"""
Entrena una red neuronal recurrente (LSTM) para detectar gestos m�ltiples
usando los tensores de esqueleto extra�dos de videos.

ESTRUCTURA DEL TENSOR:
- Cabeza: nose_x, nose_y, yaw, pitch, roll (5 valores)
- Brazos: codo, mu�eca, index de cada brazo = 6 joints x 3 coords = 18 valores
- Total: 23 features por frame

VERSI�N ROBUSTA: Modelo m�s grande, data augmentation, entrenamiento exhaustivo.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from pathlib import Path
import json
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# SEMILLA GLOBAL PARA REPRODUCIBILIDAD
# ============================================================================
SEED = 42

def set_seed(seed=SEED):
    """Fija todas las semillas para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# Importar configuraci�n de features desde process_videos
from process_videos import (
    NUM_FEATURES, NUM_ARM_JOINTS, HEAD_POSE_MULTIPLIER,
    ARM_LANDMARK_NAMES, FEATURE_NAMES
)

# Tama�o de input: NUM_FEATURES (head + arms)
INPUT_SIZE = NUM_FEATURES  # 5 + 18 = 23

# ============================================================================
# CONFIGURACI�N DE ENTRENAMIENTO ROBUSTO
# ============================================================================

TRAINING_CONFIG = {
    "hidden_size": 512,
    "num_layers": 3,
    "num_attention_heads": 8,
    "fc_hidden_sizes": [
      512,
      256,
      128
    ],
    "dropout": 0.2,
    "epochs": 500,
    "batch_size": 16,
    "learning_rate": 0.0005,
    "weight_decay": 1e-05,
    "gradient_clip": 1.0,
    "patience": 50,
    "label_smoothing": 0.05,
    "augmentation": {
      "enabled": True,
      "time_warp_prob": 0.5,
      "time_warp_factor": 0.2,
      "noise_prob": 0.6,
      "noise_std": 0.025,
      "scale_prob": 0.5,
      "scale_range": [
        0.85,
        1.15
      ],
      "rotation_prob": 0.5,
      "rotation_range": [
        -20,
        20
      ],
      "dropout_joints_prob": 0.2,
      "dropout_joints_rate": 0.1,
      "time_mask_prob": 0.2,
      "time_mask_ratio": 0.1
    },
    "scheduler": "onecycle",
    "warmup_epochs": 10,
    "T_0": 40,
    "T_mult": 2,
    "mixup_alpha": 0.15,
    "use_kfold": True,
    "n_folds": 5
  }

# Estados disponibles
STATES = [
    "no_person",
    "wave",
    "handshake_right",
    "handshake_left",
    "hug",
    "dance",
    "fall",
    "high_five",
    "yes",
    "no",
    "default"
]


class SkeletonAugmentation:
    """Data augmentation para secuencias de features (head pose + brazos)"""

    def __init__(self, config):
        self.config = config
        self.enabled = config.get('enabled', False)

    def __call__(self, tensor):
        """
        Aplica augmentations al tensor [frames, NUM_FEATURES]

        Args:
            tensor: numpy array [frames, 33] con estructura:
                    [0-2]: head pose (yaw, pitch, roll)
                    [3-32]: brazos (10 joints * 3 coords)

        Returns:
            tensor augmentado
        """
        if not self.enabled:
            return tensor

        tensor = tensor.copy()

        # Time warping (estirar/comprimir tiempo)
        if random.random() < self.config.get('time_warp_prob', 0):
            tensor = self._time_warp(tensor)

        # A�adir ruido gaussiano
        if random.random() < self.config.get('noise_prob', 0):
            tensor = self._add_noise(tensor)

        # Escalado de brazos (no head pose)
        if random.random() < self.config.get('scale_prob', 0):
            tensor = self._scale_arms(tensor)

        # Rotaci�n 2D de brazos (en plano XY)
        if random.random() < self.config.get('rotation_prob', 0):
            tensor = self._rotate_arms_2d(tensor)

        # Dropout de features aleatorias
        if random.random() < self.config.get('dropout_joints_prob', 0):
            tensor = self._dropout_features(tensor)

        # Enmascarar frames temporales
        if random.random() < self.config.get('time_mask_prob', 0):
            tensor = self._time_mask(tensor)

        return tensor

    def _time_warp(self, tensor):
        """Warping temporal: estirar o comprimir la secuencia"""
        factor = self.config.get('time_warp_factor', 0.2)
        scale = 1.0 + random.uniform(-factor, factor)

        num_frames = tensor.shape[0]
        new_num_frames = int(num_frames * scale)

        if new_num_frames < 2:
            return tensor

        # Interpolar a nueva longitud
        old_indices = np.linspace(0, num_frames - 1, new_num_frames)
        new_tensor = np.zeros((new_num_frames, tensor.shape[1]))

        for f in range(tensor.shape[1]):
            new_tensor[:, f] = np.interp(
                old_indices,
                np.arange(num_frames),
                tensor[:, f]
            )

        return new_tensor

    def _add_noise(self, tensor):
        """A�ade ruido gaussiano"""
        std = self.config.get('noise_std', 0.02)
        noise = np.random.normal(0, std, tensor.shape)
        # Solo a�adir ruido donde hay datos v�lidos (no NaN/0)
        mask = ~np.isnan(tensor) & (tensor != 0)
        tensor[mask] += noise[mask]
        return tensor

    def _scale_arms(self, tensor):
        """Escalado de coordenadas de brazos (�ndices 3-32)"""
        scale_range = self.config.get('scale_range', (0.9, 1.1))
        scale = random.uniform(scale_range[0], scale_range[1])
        # Solo escalar brazos, no head pose
        tensor[:, 3:] = tensor[:, 3:] * scale
        return tensor

    def _rotate_arms_2d(self, tensor):
        """Rotaci�n de brazos en el plano XY"""
        angle_range = self.config.get('rotation_range', (-15, 15))
        angle = random.uniform(angle_range[0], angle_range[1])
        angle_rad = np.radians(angle)

        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Rotar cada joint de brazos (10 joints, cada uno con x,y,z)
        for joint_idx in range(NUM_ARM_JOINTS):
            base_idx = 3 + joint_idx * 3  # Offset de 3 por head pose
            x_idx = base_idx
            y_idx = base_idx + 1

            for frame in range(tensor.shape[0]):
                x, y = tensor[frame, x_idx], tensor[frame, y_idx]
                if not np.isnan(x) and not np.isnan(y):
                    tensor[frame, x_idx] = cos_a * x - sin_a * y
                    tensor[frame, y_idx] = sin_a * x + cos_a * y

        return tensor

    def _dropout_features(self, tensor):
        """Dropout aleatorio de features"""
        dropout_rate = self.config.get('dropout_joints_rate', 0.1)

        # Dropout de joints de brazos completos (3 features por joint)
        num_joints = NUM_ARM_JOINTS
        num_dropout = int(num_joints * dropout_rate)

        if num_dropout > 0:
            joints_to_drop = random.sample(range(num_joints), num_dropout)
            for joint_idx in joints_to_drop:
                base_idx = 3 + joint_idx * 3
                tensor[:, base_idx:base_idx + 3] = 0.0

        return tensor

    def _time_mask(self, tensor):
        """Enmascara frames consecutivos (como SpecAugment)"""
        mask_ratio = self.config.get('time_mask_ratio', 0.1)
        num_frames = tensor.shape[0]
        mask_length = int(num_frames * mask_ratio)

        if mask_length > 0 and num_frames > mask_length:
            start = random.randint(0, num_frames - mask_length)
            tensor[start:start + mask_length, :] = 0.0

        return tensor


class SkeletonDataset(Dataset):
    """Dataset de tensores de esqueleto para PyTorch con data augmentation"""

    def __init__(self, tensor_files, labels, max_frames=None, augmentation=None, training=True):
        """
        Args:
            tensor_files: Lista de rutas a archivos .npy
            labels: Lista de labels (�ndices de clase)
            max_frames: M�ximo n�mero de frames (para padding/truncate)
            augmentation: Configuraci�n de augmentation
            training: Si es True, aplica augmentation
        """
        self.tensor_files = tensor_files
        self.labels = labels
        self.max_frames = max_frames
        self.training = training

        # Inicializar augmentation
        if augmentation is not None and training:
            self.augmentation = SkeletonAugmentation(augmentation)
        else:
            self.augmentation = None

        # Calcular max_frames autom�ticamente si no se especifica
        if self.max_frames is None:
            self.max_frames = self._calculate_max_frames()

    def _calculate_max_frames(self):
        """Calcula el n�mero m�ximo de frames en el dataset"""
        max_f = 0
        for f in self.tensor_files:
            tensor = np.load(f)
            max_f = max(max_f, tensor.shape[0])
        return max_f

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        # Cargar tensor [frames, NUM_FEATURES] - ya viene aplanado
        tensor = np.load(self.tensor_files[idx])

        # Aplicar augmentation si est� habilitado
        if self.augmentation is not None:
            tensor = self.augmentation(tensor)

        # El tensor ya viene en formato [frames, NUM_FEATURES]
        # Reemplazar NaN con 0 (features no detectadas)
        tensor_clean = np.nan_to_num(tensor, nan=0.0)

        # Padding o truncate a max_frames
        if tensor_clean.shape[0] < self.max_frames:
            # Padding con ceros
            padding = np.zeros((self.max_frames - tensor_clean.shape[0], INPUT_SIZE))
            tensor_padded = np.vstack([tensor_clean, padding])
        else:
            # Truncate
            tensor_padded = tensor_clean[:self.max_frames]

        # Convertir a tensor de PyTorch
        x = torch.FloatTensor(tensor_padded)
        y = torch.LongTensor([self.labels[idx]])  # LongTensor para CrossEntropyLoss

        # Calcular m�scara de secuencia v�lida
        valid_length = min(tensor.shape[0], self.max_frames)

        return x, y, valid_length


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention para secuencias temporales"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention con residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_out))
        return x


class GestureDetectorLSTM(nn.Module):
    """
    Red neuronal recurrente robusta para detectar gestos m�ltiples.

    Arquitectura:
    - Input projection con BatchNorm
    - Stacked Bidirectional LSTM
    - Multi-Head Self Attention
    - Deep fully connected layers con residual connections
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=256, num_layers=4,
                 dropout=0.4, num_classes=11, num_attention_heads=8,
                 fc_hidden_sizes=None):
        """
        Args:
            input_size: N�mero de features (NUM_JOINTS * 3 coords = 69)
            hidden_size: Tama�o de capa oculta LSTM
            num_layers: N�mero de capas LSTM
            dropout: Dropout rate
            num_classes: N�mero de clases a predecir
            num_attention_heads: N�mero de cabezas de atenci�n
            fc_hidden_sizes: Lista de tama�os de capas FC
        """
        super(GestureDetectorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        if fc_hidden_sizes is None:
            fc_hidden_sizes = [512, 256, 128]

        # Input projection con BatchNorm
        self.input_bn = nn.BatchNorm1d(input_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout * 0.5)

        # Stacked Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Layer normalization despu�s de LSTM
        self.lstm_ln = nn.LayerNorm(hidden_size * 2)

        # Multi-Head Self Attention
        self.self_attention = MultiHeadSelfAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Segunda capa de atenci�n para capturar patrones m�s complejos
        self.self_attention2 = MultiHeadSelfAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Temporal pooling attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Deep fully connected layers con residual connections
        fc_layers = []
        prev_size = hidden_size * 2

        for i, fc_size in enumerate(fc_hidden_sizes):
            fc_layers.append(nn.Linear(prev_size, fc_size))
            fc_layers.append(nn.LayerNorm(fc_size))
            fc_layers.append(nn.GELU())  # GELU en lugar de ReLU
            fc_layers.append(nn.Dropout(dropout))
            prev_size = fc_size

        self.fc_layers = nn.ModuleList(fc_layers)

        # Capa final de clasificaci�n
        self.classifier = nn.Linear(fc_hidden_sizes[-1], num_classes)

        # Inicializaci�n de pesos
        self._init_weights()

    def _init_weights(self):
        """Inicializaci�n Xavier/Glorot para mejor convergencia"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor [batch, seq_len, input_size]
            lengths: Longitudes v�lidas de cada secuencia

        Returns:
            output: Logits de clasificaci�n [batch, num_classes]
        """
        batch_size, seq_len, _ = x.shape

        # Input projection con BatchNorm
        x_reshaped = x.reshape(-1, x.shape[-1])  # [batch*seq, input_size]
        x_bn = self.input_bn(x_reshaped)
        x_proj = self.input_projection(x_bn)
        x_proj = self.input_dropout(x_proj)
        x_proj = x_proj.reshape(batch_size, seq_len, -1)  # [batch, seq, hidden]

        # LSTM bidireccional
        lstm_out, _ = self.lstm(x_proj)  # [batch, seq_len, hidden_size*2]
        lstm_out = self.lstm_ln(lstm_out)

        # Multi-Head Self Attention (2 capas)
        attn_out = self.self_attention(lstm_out)
        attn_out = self.self_attention2(attn_out)

        # Temporal pooling con atenci�n
        attention_weights = torch.softmax(self.temporal_attention(attn_out), dim=1)
        context = torch.sum(attention_weights * attn_out, dim=1)  # [batch, hidden_size*2]

        # Deep FC layers
        out = context
        for layer in self.fc_layers:
            out = layer(out)

        # Clasificaci�n final
        logits = self.classifier(out)

        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy con Label Smoothing para mejor generalizaci�n"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = torch.log_softmax(pred, dim=-1)

        # Crear targets suavizados
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


def mixup_data(x, y, alpha=0.2):
    """Mixup: mezcla ejemplos de entrenamiento para regularizaci�n"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function para mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class GestureDetectorTrainer:
    """Entrenador robusto para el modelo de detecci�n de gestos"""

    def __init__(self, model, config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config or TRAINING_CONFIG
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []

    def train_epoch(self, train_loader, optimizer, criterion, mixup_alpha=0.0):
        """Entrena una �poca con mixup opcional"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for x, y, lengths in pbar:
            x, y = x.to(self.device), y.to(self.device).view(-1)

            # Aplicar mixup si est� habilitado
            if mixup_alpha > 0:
                x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
                optimizer.zero_grad()
                outputs = self.model(x, lengths)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                outputs = self.model(x, lengths)
                loss = criterion(outputs, y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            gradient_clip = self.config.get('gradient_clip', 0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

            optimizer.step()

            # M�tricas
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)

            if mixup_alpha > 0:
                # Para mixup, usar y_a para calcular accuracy aproximado
                correct += (lam * (predictions == y_a).float() +
                           (1 - lam) * (predictions == y_b).float()).sum().item()
            else:
                correct += (predictions == y).sum().item()

            total += y.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader, criterion):
        """Valida el modelo"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y = x.to(self.device), y.to(self.device).view(-1)

                outputs = self.model(x, lengths)
                loss = criterion(outputs, y)

                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs=None, lr=None, patience=None):
        """
        Entrena el modelo con configuraci�n robusta

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validaci�n
            epochs: N�mero de �pocas (usa config si no se especifica)
            lr: Learning rate (usa config si no se especifica)
            patience: Paciencia para early stopping (usa config si no se especifica)
        """
        # Usar configuraci�n global si no se especifican par�metros
        epochs = epochs or self.config['epochs']
        lr = lr or self.config['learning_rate']
        patience = patience or self.config['patience']
        weight_decay = self.config.get('weight_decay', 0)
        label_smoothing = self.config.get('label_smoothing', 0)
        mixup_alpha = self.config.get('mixup_alpha', 0)

        # Criterion con label smoothing
        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer con weight decay (L2 regularization)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Scheduler avanzado
        scheduler_type = self.config.get('scheduler', 'cosine_warm_restarts')
        if scheduler_type == 'cosine_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.get('T_0', 50),
                T_mult=self.config.get('T_mult', 2),
                eta_min=lr * 0.01
            )
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr * 10,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                anneal_strategy='cos'
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5
            )

        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0

        print(f"\n{'=' * 70}")
        print("ENTRENAMIENTO ROBUSTO")
        print(f"{'=' * 70}")
        print(f"Dispositivo: {self.device}")
        print(f"�pocas: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {weight_decay}")
        print(f"Label smoothing: {label_smoothing}")
        print(f"Mixup alpha: {mixup_alpha}")
        print(f"Gradient clipping: {self.config.get('gradient_clip', 0)}")
        print(f"Scheduler: {scheduler_type}")
        print(f"Patience: {patience}")
        print(f"{'=' * 70}")

        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            print(f"\n�poca {epoch+1}/{epochs} | LR: {current_lr:.6f}")

            # Entrenar con mixup
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, mixup_alpha
            )
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validar (sin mixup)
            val_criterion = nn.CrossEntropyLoss()  # Validaci�n sin label smoothing
            val_loss, val_acc, _, _ = self.validate(val_loader, val_criterion)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Actualizar scheduler
            if scheduler_type == 'onecycle':
                pass  # OneCycleLR se actualiza en cada step
            elif scheduler_type == 'cosine_warm_restarts':
                scheduler.step()
            else:
                scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Guardar mejor modelo (por loss y por accuracy)
            improved = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                torch.save(self.model.state_dict(), 'models/best_gesture_detector.pth')
                print("\u2713 Mejor modelo guardado (loss)")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                improved = True
                torch.save(self.model.state_dict(), 'models/best_gesture_detector_acc.pth')
                print("\u2713 Mejor modelo guardado (accuracy)")

            # Early stopping
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n\u26a0 Early stopping despu�s de {epoch+1} �pocas")
                    break

            # Guardar checkpoint cada 50 �pocas
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'models/checkpoint_epoch_{epoch+1}.pth')
                print(f"\u2713 Checkpoint guardado (�poca {epoch+1})")

        # Cargar mejor modelo
        self.model.load_state_dict(torch.load('models/best_gesture_detector.pth'))
        print(f"\n{'=' * 70}")
        print("\u2713 ENTRENAMIENTO COMPLETADO")
        print(f"Mejor Val Loss: {best_val_loss:.4f}")
        print(f"Mejor Val Acc: {best_val_acc:.4f}")
        print(f"{'=' * 70}")

    def plot_history(self, save_path='models/training_history.png', smoothing_weight=0.9):
        """
        Grafica el historial de entrenamiento con smoothing profesional

        Args:
            save_path: Ruta donde guardar la imagen
            smoothing_weight: Peso para EMA (0-1). Mayor = m�s suave. Default 0.9
        """
        def exponential_moving_average(data, weight=0.9):
            """Aplica suavizado exponencial (EMA) a los datos"""
            smoothed = []
            last = data[0]
            for point in data:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

        # Aplicar smoothing
        train_losses_smooth = exponential_moving_average(self.train_losses, smoothing_weight)
        val_losses_smooth = exponential_moving_average(self.val_losses, smoothing_weight)
        train_accs_smooth = exponential_moving_average(self.train_accs, smoothing_weight)
        val_accs_smooth = exponential_moving_average(self.val_accs, smoothing_weight)

        epochs = range(1, len(self.train_losses) + 1)

        # Configurar estilo profesional
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Colores profesionales
        train_color = '#2E86AB'  # Azul
        val_color = '#E94F37'    # Rojo

        # ===== Loss =====
        # Datos originales (transparentes)
        ax1.plot(epochs, self.train_losses, color=train_color, alpha=0.2, linewidth=1)
        ax1.plot(epochs, self.val_losses, color=val_color, alpha=0.2, linewidth=1)
        # Datos suavizados (l�neas principales)
        ax1.plot(epochs, train_losses_smooth, color=train_color, linewidth=2.5, label='Train Loss')
        ax1.plot(epochs, val_losses_smooth, color=val_color, linewidth=2.5, label='Val Loss')

        ax1.set_xlabel('�poca', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('P�rdida durante Entrenamiento', fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, len(epochs))

        # ===== Accuracy =====
        # Datos originales (transparentes)
        ax2.plot(epochs, self.train_accs, color=train_color, alpha=0.2, linewidth=1)
        ax2.plot(epochs, self.val_accs, color=val_color, alpha=0.2, linewidth=1)
        # Datos suavizados (l�neas principales)
        ax2.plot(epochs, train_accs_smooth, color=train_color, linewidth=2.5, label='Train Accuracy')
        ax2.plot(epochs, val_accs_smooth, color=val_color, linewidth=2.5, label='Val Accuracy')

        ax2.set_xlabel('�poca', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Precisi�n durante Entrenamiento', fontsize=14, fontweight='bold', pad=10)
        ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, len(epochs))
        ax2.set_ylim(0, 1.05)

        # A�adir anotaciones con mejores valores
        best_val_loss_idx = np.argmin(self.val_losses)
        best_val_acc_idx = np.argmax(self.val_accs)

        ax1.annotate(f'Min: {self.val_losses[best_val_loss_idx]:.4f}',
                    xy=(best_val_loss_idx + 1, self.val_losses[best_val_loss_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color=val_color,
                    arrowprops=dict(arrowstyle='->', color=val_color, alpha=0.7))

        ax2.annotate(f'Max: {self.val_accs[best_val_acc_idx]:.4f}',
                    xy=(best_val_acc_idx + 1, self.val_accs[best_val_acc_idx]),
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=9, color=val_color,
                    arrowprops=dict(arrowstyle='->', color=val_color, alpha=0.7))

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"\u2713 Historial guardado en {save_path}")

    def evaluate(self, test_loader, class_names):
        """Eval�a el modelo en el conjunto de test"""
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, preds, labels = self.validate(test_loader, criterion)

        print("\n" + "=" * 60)
        print("EVALUACI�N FINAL")
        print("=" * 60)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Classification report
        preds = np.array(preds)
        labels = np.array(labels)

        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=class_names, zero_division=0))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(labels, preds)
        print(cm)

        # Guardar confusion matrix como imagen
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)

        # A�adir valores en las celdas
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=150)
        plt.close()
        print("\n\u2713 Matriz de confusi�n guardada en models/confusion_matrix.png")


def load_dataset(tensors_folder='training-tensors'):
    """
    Carga el dataset de tensores desde training-tensors/

    Args:
        tensors_folder: Carpeta con los tensors

    Returns:
        tensor_files: Lista de rutas a archivos .npy
        labels: Lista de labels (�ndices de clase)
        class_names: Lista de nombres de clases (solo clases presentes)
    """
    tensors_path = Path(tensors_folder)

    if not tensors_path.exists():
        raise ValueError(f"Carpeta {tensors_folder}/ no encontrada. Ejecuta process_videos.py primero.")

    # Buscar todos los archivos de tensors
    tensor_files = list(tensors_path.glob("*_tensor.npy"))

    if len(tensor_files) == 0:
        raise ValueError(f"No se encontraron tensores en {tensors_folder}/")

    # Primero, identificar qu� estados est�n presentes
    states_found = set()
    file_to_state = {}

    for f in tensor_files:
        file_name = f.name
        state = None
        for s in STATES:
            if file_name.startswith(s + "_"):
                state = s
                break

        if state is None:
            print(f"\u26a0 Advertencia: No se pudo determinar el estado de {file_name}")
            continue

        states_found.add(state)
        file_to_state[f] = state

    # Crear lista de clases presentes en orden original
    class_names = [s for s in STATES if s in states_found]

    if len(class_names) == 0:
        raise ValueError(f"No se encontraron estados v�lidos en {tensors_folder}/")

    # Crear mapeo de estado a �ndice (solo para estados presentes)
    state_to_idx = {state: idx for idx, state in enumerate(class_names)}

    # Extraer labels usando el nuevo mapeo
    labels = []
    class_counts = {state: 0 for state in class_names}
    valid_files = []

    for f in tensor_files:
        if f not in file_to_state:
            continue

        state = file_to_state[f]
        label_idx = state_to_idx[state]
        labels.append(label_idx)
        valid_files.append(f)
        class_counts[state] += 1

    print(f"\nDataset cargado:")
    print(f"Total videos: {len(labels)}")
    print(f"Clases encontradas: {len(class_names)}")
    print("\nDistribuci�n por clase:")
    for state, count in class_counts.items():
        print(f"  {state}: {count} videos")

    return valid_files, labels, class_names


def main():
    """Funci�n principal de entrenamiento robusto con K-Fold CV opcional"""

    # Resetear semilla para garantizar reproducibilidad
    set_seed(SEED)

    # Crear carpeta de modelos
    Path('models').mkdir(exist_ok=True)

    print("=" * 70)
    print("ENTRENAMIENTO ROBUSTO DE DETECTOR DE GESTOS MULTI-CLASE")
    print(f"Semilla para reproducibilidad: {SEED}")
    print("=" * 70)

    # Mostrar configuraci�n
    print("\nConfiguraci�n de entrenamiento:")
    print(f"  Hidden size: {TRAINING_CONFIG['hidden_size']}")
    print(f"  LSTM layers: {TRAINING_CONFIG['num_layers']}")
    print(f"  Attention heads: {TRAINING_CONFIG['num_attention_heads']}")
    print(f"  FC layers: {TRAINING_CONFIG['fc_hidden_sizes']}")
    print(f"  Dropout: {TRAINING_CONFIG['dropout']}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  K-Fold CV: {TRAINING_CONFIG.get('use_kfold', False)}")
    if TRAINING_CONFIG.get('use_kfold', False):
        print(f"  N Folds: {TRAINING_CONFIG.get('n_folds', 5)}")

    # Cargar dataset
    print("\n1. Cargando dataset...")
    tensor_files, labels, class_names = load_dataset()
    tensor_files = np.array(tensor_files)
    labels = np.array(labels)

    # Verificar que hay datos
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        print(f"\n\u26a0 Advertencia: Solo se encontr� {len(unique_labels)} clase(s)")
        return

    num_classes = len(class_names)
    batch_size = TRAINING_CONFIG['batch_size']

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # ========================================================================
    # K-FOLD CROSS-VALIDATION
    # ========================================================================
    if TRAINING_CONFIG.get('use_kfold', False):
        n_folds = TRAINING_CONFIG.get('n_folds', 5)
        print(f"\n{'='*70}")
        print(f"USANDO {n_folds}-FOLD CROSS-VALIDATION")
        print(f"Cada fold eval�a ~{len(labels)//n_folds} videos")
        print(f"{'='*70}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

        fold_results = []
        all_preds = []
        all_labels_list = []

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(tensor_files, labels)):
            print(f"\n{'='*70}")
            print(f"FOLD {fold+1}/{n_folds}")
            print(f"{'='*70}")

            set_seed(SEED + fold)

            # Split train_val en train y val (85/15)
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.15, random_state=SEED,
                stratify=labels[train_val_idx]
            )

            train_files_fold = tensor_files[train_idx].tolist()
            train_labels_fold = labels[train_idx].tolist()
            val_files_fold = tensor_files[val_idx].tolist()
            val_labels_fold = labels[val_idx].tolist()
            test_files_fold = tensor_files[test_idx].tolist()
            test_labels_fold = labels[test_idx].tolist()

            print(f"  Train: {len(train_files_fold)}, Val: {len(val_files_fold)}, Test: {len(test_files_fold)}")

            # Crear datasets
            train_dataset = SkeletonDataset(
                train_files_fold, train_labels_fold, max_frames=300,
                augmentation=TRAINING_CONFIG['augmentation'], training=True
            )
            val_dataset = SkeletonDataset(
                val_files_fold, val_labels_fold, max_frames=300,
                augmentation=None, training=False
            )
            test_dataset = SkeletonDataset(
                test_files_fold, test_labels_fold, max_frames=300,
                augmentation=None, training=False
            )

            g = torch.Generator()
            g.manual_seed(SEED + fold)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True,
                worker_init_fn=seed_worker, generator=g
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

            # Crear modelo nuevo para cada fold
            model = GestureDetectorLSTM(
                input_size=INPUT_SIZE,
                hidden_size=TRAINING_CONFIG['hidden_size'],
                num_layers=TRAINING_CONFIG['num_layers'],
                dropout=TRAINING_CONFIG['dropout'],
                num_classes=num_classes,
                num_attention_heads=TRAINING_CONFIG['num_attention_heads'],
                fc_hidden_sizes=TRAINING_CONFIG['fc_hidden_sizes']
            )

            # Entrenar
            trainer = GestureDetectorTrainer(model, config=TRAINING_CONFIG)
            trainer.train(train_loader, val_loader)

            # Evaluar
            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc, preds, labels_fold = trainer.validate(test_loader, criterion)

            fold_results.append({'fold': fold+1, 'test_acc': test_acc, 'test_loss': test_loss})
            all_preds.extend(preds)
            all_labels_list.extend(labels_fold)

            print(f"\nFold {fold+1} - Test Accuracy: {test_acc:.4f}")

        # Resultados finales de Cross-Validation
        print(f"\n{'='*70}")
        print("RESULTADOS K-FOLD CROSS-VALIDATION")
        print(f"{'='*70}")

        accs = [r['test_acc'] for r in fold_results]
        print(f"\nAccuracy por fold:")
        for r in fold_results:
            print(f"  Fold {r['fold']}: {r['test_acc']:.4f}")

        print(f"\nAccuracy promedio: {np.mean(accs):.4f} � {np.std(accs):.4f}")
        print(f"Accuracy m�nimo: {np.min(accs):.4f}")
        print(f"Accuracy m�ximo: {np.max(accs):.4f}")

        # ================================================================
        # EVALUACI�N FINAL AGREGADA DE TODOS LOS FOLDS
        # ================================================================
        print(f"\n{'='*60}")
        print("EVALUACI�N FINAL (K-Fold Cross-Validation)")
        print("="*60)
        print(f"Total de videos evaluados: {len(all_labels_list)}")
        print(f"Accuracy promedio: {np.mean(accs):.4f} � {np.std(accs):.4f}")

        print("\nClassification Report:")
        print(classification_report(all_labels_list, all_preds, target_names=class_names, zero_division=0))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels_list, all_preds)
        print(cm)

        # Guardar confusion matrix como imagen
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (K-Fold CV)')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)

        # A�adir valores en las celdas
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix_kfold.png', dpi=150)
        plt.close()
        print("\n\u2713 Matriz de confusi�n guardada en models/confusion_matrix_kfold.png")

        # Entrenar modelo final con todos los datos para guardar
        print(f"\n{'='*70}")
        print("ENTRENANDO MODELO FINAL CON TODOS LOS DATOS")
        print(f"{'='*70}")

        set_seed(SEED)
        train_files_final, val_files_final, train_labels_final, val_labels_final = train_test_split(
            tensor_files.tolist(), labels.tolist(), test_size=0.15, random_state=SEED, stratify=labels
        )

        train_dataset = SkeletonDataset(
            train_files_final, train_labels_final, max_frames=300,
            augmentation=TRAINING_CONFIG['augmentation'], training=True
        )
        val_dataset = SkeletonDataset(
            val_files_final, val_labels_final, max_frames=300,
            augmentation=None, training=False
        )

        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, generator=g
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        model = GestureDetectorLSTM(
            input_size=INPUT_SIZE,
            hidden_size=TRAINING_CONFIG['hidden_size'],
            num_layers=TRAINING_CONFIG['num_layers'],
            dropout=TRAINING_CONFIG['dropout'],
            num_classes=num_classes,
            num_attention_heads=TRAINING_CONFIG['num_attention_heads'],
            fc_hidden_sizes=TRAINING_CONFIG['fc_hidden_sizes']
        )

        trainer = GestureDetectorTrainer(model, config=TRAINING_CONFIG)
        trainer.train(train_loader, val_loader)
        trainer.plot_history()

    # ========================================================================
    # ENTRENAMIENTO NORMAL (sin K-Fold)
    # ========================================================================
    else:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            tensor_files.tolist(), labels.tolist(), test_size=0.3, random_state=SEED, stratify=labels
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
        )

        print(f"\nSplit del dataset:")
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        train_dataset = SkeletonDataset(
            train_files, train_labels, max_frames=300,
            augmentation=TRAINING_CONFIG['augmentation'], training=True
        )
        val_dataset = SkeletonDataset(
            val_files, val_labels, max_frames=300, augmentation=None, training=False
        )
        test_dataset = SkeletonDataset(
            test_files, test_labels, max_frames=300, augmentation=None, training=False
        )

        g = torch.Generator()
        g.manual_seed(SEED)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, generator=g
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        model = GestureDetectorLSTM(
            input_size=INPUT_SIZE,
            hidden_size=TRAINING_CONFIG['hidden_size'],
            num_layers=TRAINING_CONFIG['num_layers'],
            dropout=TRAINING_CONFIG['dropout'],
            num_classes=num_classes,
            num_attention_heads=TRAINING_CONFIG['num_attention_heads'],
            fc_hidden_sizes=TRAINING_CONFIG['fc_hidden_sizes']
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nPar�metros totales: {num_params:,}")
        print(f"N�mero de clases: {num_classes}")

        trainer = GestureDetectorTrainer(model, config=TRAINING_CONFIG)
        trainer.train(train_loader, val_loader)
        trainer.evaluate(test_loader, class_names)
        trainer.plot_history()

    # Guardar configuraci�n completa del modelo
    num_params = sum(p.numel() for p in model.parameters())
    config = {
        'input_size': INPUT_SIZE,
        'hidden_size': TRAINING_CONFIG['hidden_size'],
        'num_layers': TRAINING_CONFIG['num_layers'],
        'dropout': TRAINING_CONFIG['dropout'],
        'num_attention_heads': TRAINING_CONFIG['num_attention_heads'],
        'fc_hidden_sizes': TRAINING_CONFIG['fc_hidden_sizes'],
        'max_frames': 300,
        'num_params': num_params,
        'num_classes': num_classes,
        'classes': class_names,
        'num_features': NUM_FEATURES,
        'num_arm_joints': NUM_ARM_JOINTS,
        'head_pose_multiplier': HEAD_POSE_MULTIPLIER,
        'feature_names': FEATURE_NAMES,
        'arm_landmark_names': ARM_LANDMARK_NAMES,
        'training_config': TRAINING_CONFIG
    }

    with open('models/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print("Archivos guardados:")
    print("  - models/best_gesture_detector.pth (mejor loss)")
    print("  - models/best_gesture_detector_acc.pth (mejor accuracy)")
    print("  - models/model_config.json")
    print("  - models/training_history.png")
    print("\nPara usar el modelo, ejecuta: python realtime_gesture_detector.py")


if __name__ == '__main__':
    main()
