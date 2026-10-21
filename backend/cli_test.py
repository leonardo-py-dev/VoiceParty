import os
import sys
import argparse
import glob

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.core.scorer import VoiceScorer
from backend.ml.train import train_siamese_network

def print_result_card(res, title="RESULTADO DA COMPARAÇÃO"):
    print("\n" + "=" * 50)
    print(f"       🎤 {title} 🎤")
    print("=" * 50)
    print(f" ⭐ NOTA GERAL:           {res.total_score} / 100")
    print(f" 🧠 Similaridade Neural:   {res.neural_score}% (Rede Siamesa CRNN)")
    print(f" 🎵 Entonação / Pitch:     {res.pitch_score}% (Contorno melódico DTW)")
    print(f" ⏱️ Ritmo / Cadência:      {res.rhythm_score}% (Envelope de energia RMS)")
    if res.word_score is not None:
        print(f" 💬 Fidelidade Palavras:   {res.word_score}%")
    print("-" * 50)
    print(f" 📢 Feedback da IA: {res.feedback}")
    print("=" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="VoiceParty - CLI de Testes de IA e Rede Neural")
    subparsers = parser.add_subparsers(dest="command")

    # Comando: compare
    compare_parser = subparsers.add_parser("compare", help="Compara dois arquivos de áudio")
    compare_parser.add_argument("ref", help="Caminho do áudio original de referência")
    compare_parser.add_argument("user", help="Caminho da tentativa gravada pelo usuário")
    compare_parser.add_argument("--ref-text", help="Texto do áudio original (opcional)", default=None)
    compare_parser.add_argument("--user-text", help="Texto dito pelo usuário (opcional)", default=None)

    # Comando: test-demo
    subparsers.add_parser("test-demo", help="Executa testes automáticos comparando o meme 'birl' com boa e má imitação")

    # Comando: train
    train_parser = subparsers.add_parser("train", help="Treina a Rede Neural Siamesa")
    train_parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de treino (padrão: 10)")

    args = parser.parse_args()

    if args.command == "compare":
        scorer = VoiceScorer()
        res = scorer.evaluate_imitation(args.ref, args.user, args.ref_text, args.user_text)
        print_result_card(res)

    elif args.command == "test-demo":
        clips_dir = os.path.join(os.path.dirname(__file__), "data", "clips")
        ref_path = os.path.join(clips_dir, "birl.wav")
        good_path = os.path.join(clips_dir, "test_attempt_good.wav")
        bad_path = os.path.join(clips_dir, "test_attempt_bad.wav")

        if not os.path.exists(ref_path):
            print("❌ Arquivos de demonstração não encontrados. Execute 'python backend/generate_samples.py' primeiro.")
            return

        scorer = VoiceScorer()
        print("\n🧪 1. TESTANDO IMITAÇÃO BOA (Mesma entonação e ritmo próximo de 'birl.wav'):")
        res_good = scorer.evaluate_imitation(ref_path, good_path)
        print_result_card(res_good, "TESTE: BOA IMITAÇÃO")

        print("\n🧪 2. TESTANDO IMITAÇÃO RUIM (Fala monótona/desafinada):")
        res_bad = scorer.evaluate_imitation(ref_path, bad_path)
        print_result_card(res_bad, "TESTE: MÁ IMITAÇÃO")

    elif args.command == "train":
        clips_dir = os.path.join(os.path.dirname(__file__), "data", "clips")
        audio_files = glob.glob(os.path.join(clips_dir, "*.wav")) + glob.glob(os.path.join(clips_dir, "*.mp3"))
        train_siamese_network(audio_files, epochs=args.epochs)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
