import os
import sys
import ssl
import re
import json
import urllib.request
import urllib.parse
import subprocess
import imageio_ffmpeg
import soundfile as sf
import librosa

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Desativa verificação rígida de certificados para downloads em scripts locais
ssl._create_default_https_context = ssl._create_unverified_context

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
CLIPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "clips"))
os.makedirs(CLIPS_DIR, exist_ok=True)
METADATA_PATH = os.path.join(CLIPS_DIR, "metadata.json")

# Lista de memes e falas clássicas com seus slugs/buscas no MyInstants
MEMES_CONFIG = [
    {
        "id": "birl",
        "title": "Birl! É Treze!",
        "phrase": "Biiiiirl! Não vai dá não! Que não vai dá o que, rapaz! Sai de casa, comi pra caralho!",
        "urls": [
            "https://www.myinstants.com/media/sounds/birl.mp3",
            "https://www.myinstants.com/media/sounds/birl-original.mp3"
        ],
        "search": "birl"
    },
    {
        "id": "ta_saindo_da_jaula",
        "title": "Tá Saindo da Jaula o Monstro",
        "phrase": "Tá saindo da jaula o monstro, porra!",
        "urls": [
            "https://www.myinstants.com/media/sounds/ta-saindo-da-jaula-o-monstro.mp3",
            "https://www.myinstants.com/media/sounds/ta-saindo-da-jaula-o-monstro-bambam.mp3"
        ],
        "search": "ta saindo da jaula o monstro"
    },
    {
        "id": "pede_pra_sair",
        "title": "Tropa de Elite - Pede pra Sair",
        "phrase": "Pede pra sair! Pede pra sair, zero um!",
        "urls": [
            "https://www.myinstants.com/media/sounds/pede-pra-sair.mp3",
            "https://www.myinstants.com/media/sounds/capitao-nascimento-pede-pra-sair.mp3"
        ],
        "search": "pede pra sair"
    },
    {
        "id": "ze_pequeno",
        "title": "Cidade de Deus - Zé Pequeno",
        "phrase": "Meu nome agora é Zé Pequeno, porra!",
        "urls": [
            "https://www.myinstants.com/media/sounds/meu-nome-agora-e-ze-pequeno-porra.mp3",
            "https://www.myinstants.com/media/sounds/meu-nome-e-ze-pequeno.mp3"
        ],
        "search": "meu nome agora e ze pequeno"
    },
    {
        "id": "ola_marilene",
        "title": "Olá Marilene",
        "phrase": "Olá Marilene, a noite é uma criança!",
        "urls": [
            "https://www.myinstants.com/media/sounds/ola-marilene.mp3",
            "https://www.myinstants.com/media/sounds/ola-marilene-original.mp3"
        ],
        "search": "ola marilene"
    },
    {
        "id": "ja_acabou_jessica",
        "title": "Já Acabou, Jéssica?",
        "phrase": "Já acabou, Jéssica?",
        "urls": [
            "https://www.myinstants.com/media/sounds/ja-acabou-jessica.mp3",
            "https://www.myinstants.com/media/sounds/ja-acabou-jessica-original.mp3"
        ],
        "search": "ja acabou jessica"
    },
    {
        "id": "receba",
        "title": "Luva de Pedreiro - Receba!",
        "phrase": "Receba! Graças a Deus, pai! O melhor do mundo!",
        "urls": [
            "https://www.myinstants.com/media/sounds/receba-luva-de-pedreiro.mp3",
            "https://www.myinstants.com/media/sounds/receba.mp3"
        ],
        "search": "receba luva de pedreiro"
    },
    {
        "id": "calma_calabreso",
        "title": "Calma, Calabreso!",
        "phrase": "Calma, calabreso! Relaxa!",
        "urls": [
            "https://www.myinstants.com/media/sounds/calma-calabreso.mp3"
        ],
        "search": "calma calabreso"
    }
]

def search_myinstants_mp3(query: str):
    """Busca o link direto do MP3 na página de busca do MyInstants se os links diretos falharem."""
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.myinstants.com/pt/search/?name={encoded_query}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            # Procura por /media/sounds/....mp3
            matches = re.findall(r'/media/sounds/[^"\'<>]+\.mp3', html)
            if matches:
                return f"https://www.myinstants.com{matches[0]}"
    except Exception as e:
        print(f"  Erro na busca por '{query}': {e}")
    return None

def download_and_convert(mp3_url: str, output_wav_path: str) -> bool:
    """Baixa o MP3 e converte para WAV 16kHz Mono limpo usando ffmpeg."""
    temp_mp3 = output_wav_path + ".temp.mp3"
    try:
        req = urllib.request.Request(mp3_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            if len(content) < 1000:  # Muito pequeno para ser áudio válido
                return False
            with open(temp_mp3, "wb") as f:
                f.write(content)

        # Converte para WAV 16kHz mono com ffmpeg
        cmd = [
            FFMPEG_EXE, "-y", "-i", temp_mp3,
            "-ar", "16000", "-ac", "1",
            output_wav_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(output_wav_path):
            # Valida com soundfile
            info = sf.info(output_wav_path)
            if info.duration > 0.5:
                return True
        return False
    except Exception as e:
        return False
    finally:
        if os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
            except:
                pass

def main():
    print("🎙️ Baixando os áudios originais autênticos dos memes e falas...")
    updated_metadata = []

    for meme in MEMES_CONFIG:
        target_wav = os.path.join(CLIPS_DIR, f"{meme['id']}.wav")
        print(f"\n🔎 Buscando áudio original: {meme['title']}...")

        downloaded = False
        # Tenta URLs diretas conhecidas
        for url in meme.get("urls", []):
            if download_and_convert(url, target_wav):
                downloaded = True
                print(f"  ✅ Baixado com sucesso via link direto: {url}")
                break

        # Se não baixou, busca no MyInstants
        if not downloaded:
            scraped_url = search_myinstants_mp3(meme["search"])
            if scraped_url and download_and_convert(scraped_url, target_wav):
                downloaded = True
                print(f"  ✅ Baixado com sucesso via busca: {scraped_url}")

        if downloaded and os.path.exists(target_wav):
            duration = round(sf.info(target_wav).duration, 2)
            updated_metadata.append({
                "id": meme["id"],
                "title": meme["title"],
                "phrase": meme["phrase"],
                "file": f"{meme['id']}.wav",
                "duration": duration
            })
            print(f"  🔊 Duração: {duration}s | Frase: \"{meme['phrase']}\"")
        else:
            print(f"  ⚠️ Não foi possível obter o áudio para '{meme['title']}'.")

    if updated_metadata:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
        print(f"\n🎉 Sucesso! {len(updated_metadata)} áudios originais autênticos foram instalados em 'backend/data/clips/'.")
    else:
        print("\n❌ Nenhum áudio pôde ser baixado.")

if __name__ == "__main__":
    main()
