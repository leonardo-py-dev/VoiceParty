import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.ml.model import SiameseAudioCRNN
from backend.ml.dataset import AudioTripletDataset

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "siamese_crnn.pt")

def train_siamese_network(
    audio_paths: list[str],
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.001,
    margin: float = 0.3
):
    print(f"\n🚀 Iniciando treinamento da Rede Neural Siamesa...")
    print(f"📊 Áudios base carregados: {len(audio_paths)}")
    print(f"⚙️ Configuração: {epochs} épocas, batch size {batch_size}, Triplet Loss (margin={margin})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Dispositivo de treino: {device}")

    # Dataset & DataLoader
    dataset = AudioTripletDataset(audio_paths=audio_paths, samples_per_epoch=240)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modelo, Otimizador e Custo
    model = SiameseAudioCRNN(n_mels=64, embedding_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2.0)

    loss_history = []
    best_loss = float("inf")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0

        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Extração de embeddings dos três ramos da rede siamesa
            emb_anchor = model.forward_one(anchor)
            emb_pos = model.forward_one(positive)
            emb_neg = model.forward_one(negative)

            # Triplet Loss: força d(anchor, positive) < d(anchor, negative) - margin
            loss = triplet_loss(emb_anchor, emb_pos, emb_neg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        loss_history.append(avg_loss)
        print(f"  [Época {epoch:02d}/{epochs:02d}] - Loss Triplet: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
            }, CHECKPOINT_PATH)

    print(f"\n✅ Treinamento concluído com sucesso!")
    print(f"💾 Melhor modelo salvo em: {CHECKPOINT_PATH}")

    # Salva gráfico da curva de perda para apresentação / relatórios
    try:
        plot_path = os.path.join(CHECKPOINT_DIR, "loss_curve.png")
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, epochs + 1), loss_history, marker='o', color='#3b82f6', label='Triplet Loss')
        plt.title("Evolução do Treinamento da Rede Siamesa (VoiceParty)")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Gráfico da curva de perda salvo em: {plot_path}")
    except Exception as e:
        print(f"Nota: não foi possível salvar o gráfico: {e}")

    return model

if __name__ == "__main__":
    clips_dir = os.path.join(os.path.dirname(__file__), "..", "data", "clips")
    audio_files = glob.glob(os.path.join(clips_dir, "*.wav")) + glob.glob(os.path.join(clips_dir, "*.mp3"))

    if not audio_files:
        print("❌ Nenhum arquivo de áudio encontrado em 'backend/data/clips/'. Execute primeiro o gerador de exemplos.")
    else:
        train_siamese_network(audio_files, epochs=10)
