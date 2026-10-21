import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_size=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.drop = nn.Dropout2d(0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        return x

class SiameseAudioCRNN(nn.Module):
    """
    Rede Neural Siamesa (Convolutional-Recurrent Neural Network - CRNN)
    para comparar a similaridade acústica e expressiva entre dois áudios (Original vs. Imitação).
    
    Entrada: Espectrograma Log-Mel [Batch, 1, n_mels, time_frames]
    Saída: Vetor de Embedding normalizado L2 de 128 dimensões.
    """
    def __init__(self, n_mels: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim

        # Blocos Convolucionais para padrões de formantes e timbre
        self.block1 = ConvBlock(1, 32, pool_size=(2, 2))
        self.block2 = ConvBlock(32, 64, pool_size=(2, 2))
        self.block3 = ConvBlock(64, 128, pool_size=(2, 2))

        # Após 3 pools (2x2), n_mels (64) -> 64 / 8 = 8 frequências residuais
        # Cada frame terá 128 canais * 8 freq = 1024 dimensões
        conv_out_features = 128 * (n_mels // 8)

        # Camada Recorrente Bidirecional (BiGRU) para capturar a evolução da fala no tempo
        self.gru = nn.GRU(
            input_size=conv_out_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Projeção final para o espaço de embedding
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recebe um espectrograma [B, 1, n_mels, T] e retorna o embedding [B, embedding_dim] normalizado L2.
        """
        # Formato de entrada: [B, 1, freq, time]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)  # [B, channels=128, freq_reduced=8, time_reduced]

        b, c, f, t = x.size()
        # Remodelar para sequência temporal: [B, time_reduced, channels * freq_reduced]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Passagem pela GRU
        gru_out, _ = self.gru(x)  # [B, t, 128]

        # Temporal Global Average Pooling + Max Pooling combinados
        avg_pool = torch.mean(gru_out, dim=1)
        max_pool, _ = torch.max(gru_out, dim=1)
        pooled = (avg_pool + max_pool) / 2.0

        # Projeção para embedding
        embedding = self.fc(pooled)

        # Normalização L2 (faz com que o comprimento do vetor seja 1.0)
        # Isso permite calcular a similaridade diretamente por produto escalar / cosseno
        normalized_emb = F.normalize(embedding, p=2, dim=-1)
        return normalized_emb

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """Passa dois áudios pela mesma rede compartilhada."""
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

    @staticmethod
    def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Calcula a similaridade de cosseno mapeada para a escala de 0.0 (totalmente diferente) a 1.0 (idêntico).
        """
        # Como os vetores já possuem norma 1 (L2), o cosseno é simplesmente o produto escalar
        cosine = torch.sum(emb1 * emb2, dim=-1)
        # Mapeia [-1, 1] para [0, 1]
        similarity = torch.clamp((cosine + 1.0) / 2.0, 0.0, 1.0)
        return similarity
