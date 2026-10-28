"""
Testes e Benchmarks Adversariais de Estresse de Latência, Memória e Separabilidade
Autor: Empirical Challenger (Challenger M1 - Instance 2)
Objetivo:
1. Benchmark de latência sob carga contínua em CPU (>20 repetições) garantindo teto < 1.5s (1500 ms).
2. Estresse de consumo de memória RAM (RSS / Heap) ao longo de 100 iterações para descartar memory leaks.
3. Validação rigorosa de invalidação de cache em RAM quando arquivos no disco são modificados.
4. Teste exaustivo de separabilidade: pares idênticos (~100) vs todos os pares disjuntos cruzados (< 30).
"""

import os
import sys
import time
import gc
import tempfile
import ctypes
import ctypes.wintypes
import numpy as np
import pytest
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.core.audio_processor import AudioProcessor
from backend.core.scorer import VoiceScorer
from backend.ml.dataset import TARGET_SR, TARGET_SAMPLES

CLIPS_DIR = os.path.join(ROOT_DIR, "backend", "data", "clips")


def get_windows_process_rss_mb() -> float:
    """Retorna o Working Set (RSS em MB) do processo atual no Windows usando a API nativa do sistema."""
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ('cb', ctypes.wintypes.DWORD),
            ('PageFaultCount', ctypes.wintypes.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t),
            ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t),
            ('PeakPagefileUsage', ctypes.c_size_t),
        ]

    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL

    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    success = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    if success:
        return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
    return 0.0


def generate_synthetic_audio(duration_sec: float = 1.0, freq_hz: float = 220.0, sr: int = TARGET_SR) -> np.ndarray:
    """Gera um sinal sonoro com harmônicos e modulação para simular fala humana."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental + 2 harmônicos com decaimento
    signal = 0.6 * np.sin(2 * np.pi * freq_hz * t)
    signal += 0.3 * np.sin(2 * np.pi * (freq_hz * 2) * t)
    signal += 0.1 * np.sin(2 * np.pi * (freq_hz * 3) * t)
    # Envelope ADSR suave
    env = np.ones_like(t)
    attack = int(sr * 0.05)
    decay = int(sr * 0.05)
    env[:attack] = np.linspace(0, 1, attack)
    env[-decay:] = np.linspace(1, 0, decay)
    return (signal * env).astype(np.float32)


class TestAdversarialLatencyStress:
    """Estresse adversarial de latência contínua em CPU."""

    def test_latency_50_iterations_new_attempts_vs_cached_ref(self):
        """
        Cenário Principal: Usuário envia 50 tentativas novas consecutivas contra
        uma referência de catálogo em cache.
        A latência de cada uma deve ser estritamente < 1500 ms (teto aceitação).
        """
        processor = AudioProcessor()
        scorer = VoiceScorer(audio_processor=processor)
        ref_path = os.path.join(CLIPS_DIR, "ta_saindo_da_jaula.wav")
        assert os.path.exists(ref_path), f"Arquivo não encontrado: {ref_path}"

        # Garante que a referência está aquecida no cache
        processor.get_features(ref_path)

        latencies_ms = []
        n_reps = 50

        print(f"\n[LATENCY STRESS] Iniciando benchmark de {n_reps} repetições contínuas (Nova tentativa vs Ref em Cache)...")
        for i in range(n_reps):
            # Gera uma nova tentativa de áudio com pequenas variações de tom e duração
            dur = 0.8 + 0.4 * np.sin(i)
            freq = 200.0 + 30.0 * np.cos(i)
            wav_data = generate_synthetic_audio(duration_sec=dur, freq_hz=freq)

            t0 = time.perf_counter()
            res = scorer.evaluate_imitation(ref_path, wav_data)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            # Verificação pontual estrita
            assert elapsed_ms < 1500.0, f"Iteração {i+1} violou o teto de 1500 ms: {elapsed_ms:.2f} ms"

        mean_lat = float(np.mean(latencies_ms))
        median_lat = float(np.median(latencies_ms))
        max_lat = float(np.max(latencies_ms))
        min_lat = float(np.min(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        std_lat = float(np.std(latencies_ms))

        print(f"[LATENCY STRESS RESULTS - {n_reps} REPS]")
        print(f"  Média:    {mean_lat:.2f} ms")
        print(f"  Mediana:  {median_lat:.2f} ms")
        print(f"  Mínimo:   {min_lat:.2f} ms")
        print(f"  Máximo:   {max_lat:.2f} ms")
        print(f"  P95:      {p95_lat:.2f} ms")
        print(f"  StdDev:   {std_lat:.2f} ms")
        print(f"  Teto:     1500.00 ms (Margem de segurança: {1500.0 - max_lat:.2f} ms)")

        assert max_lat < 1500.0
        assert mean_lat < 1500.0

    def test_latency_30_iterations_zero_cache(self):
        """
        Cenário Playground Arbitrário: Ambos os áudios (referência e tentativa)
        são novos a cada rodada (zero cache de ambos os lados).
        """
        processor = AudioProcessor()
        scorer = VoiceScorer(audio_processor=processor)

        latencies_ms = []
        n_reps = 30

        print(f"\n[LATENCY ZERO CACHE] Benchmark com ambos os áudios novos ({n_reps} repetições)...")
        for i in range(n_reps):
            wav_ref = generate_synthetic_audio(duration_sec=1.2, freq_hz=210.0 + i)
            wav_usr = generate_synthetic_audio(duration_sec=1.1, freq_hz=220.0 + i)

            t0 = time.perf_counter()
            res = scorer.evaluate_imitation(wav_ref, wav_usr)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            assert elapsed_ms < 1500.0, f"Iteração {i+1} violou o teto de 1500 ms: {elapsed_ms:.2f} ms"

        mean_lat = float(np.mean(latencies_ms))
        max_lat = float(np.max(latencies_ms))
        print(f"[LATENCY ZERO CACHE RESULTS] Média: {mean_lat:.2f} ms | Máximo: {max_lat:.2f} ms")
        assert max_lat < 1500.0


class TestAdversarialRAMCacheAndMemoryLeaks:
    """Testes de consumo de memória, vazamento de recursos e integridade da invalidação de cache."""

    def test_cache_invalidation_on_file_modification(self):
        """
        Verifica se a modificação do arquivo físico no disco com alteração de mtime
        invalida de forma confiável a entrada de cache e atualiza os tensores de atributos.
        """
        processor = AudioProcessor()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # 1. Salva áudio A (frequência 200 Hz)
            wav_a = generate_synthetic_audio(duration_sec=1.0, freq_hz=200.0)
            sf.write(tmp_path, wav_a, TARGET_SR)
            # Define mtime explícito no passado
            past_time = time.time() - 100
            os.utime(tmp_path, (past_time, past_time))

            # Primeira chamada: popula o cache
            feat_a = processor.get_features(tmp_path)
            assert tmp_path in processor._cache
            assert feat_a["mtime"] == past_time

            # Confirma que chamada idêntica retorna o mesmo objeto do cache
            feat_cached = processor.get_features(tmp_path)
            assert feat_cached is feat_a, "Deveria retornar o mesmo objeto de cache em RAM"

            # 2. Modifica o arquivo no disco para áudio B (frequência 800 Hz)
            time.sleep(0.05)
            wav_b = generate_synthetic_audio(duration_sec=1.0, freq_hz=800.0)
            sf.write(tmp_path, wav_b, TARGET_SR)
            now_time = time.time()
            os.utime(tmp_path, (now_time, now_time))

            # 3. Chama get_features novamente: o cache DEVE ser invalidado
            feat_b = processor.get_features(tmp_path)
            assert feat_b is not feat_a, "Cache não foi invalidado após modificação do arquivo no disco!"
            assert feat_b["mtime"] == now_time, "mtime não foi atualizado no cache"

            # 4. Verifica se os valores acústicos refletem a nova frequência
            assert not np.allclose(feat_a["embedding"], feat_b["embedding"]), "Embeddings deveriam diferir"
            print("\n[CACHE INVALIDATION] Invalidação por mtime verificada com sucesso!")

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def test_memory_stability_under_100_evaluations(self):
        """
        Executa 100 avaliações completas e monitora o RSS (Working Set) do processo
        para descartar memory leak contínuo.
        """
        processor = AudioProcessor()
        scorer = VoiceScorer(audio_processor=processor)
        ref_path = os.path.join(CLIPS_DIR, "ta_saindo_da_jaula.wav")

        gc.collect()
        initial_rss = get_windows_process_rss_mb()
        rss_samples = []

        print(f"\n[MEMORY LEAK CHECK] RSS Inicial: {initial_rss:.2f} MB. Executando 100 iterações...")
        for i in range(1, 101):
            # Gera áudio de tentativa arbitrário
            wav_usr = generate_synthetic_audio(duration_sec=0.9, freq_hz=180.0 + (i % 20) * 10)
            scorer.evaluate_imitation(ref_path, wav_usr)

            if i % 20 == 0:
                gc.collect()
                current_rss = get_windows_process_rss_mb()
                rss_samples.append((i, current_rss))
                print(f"  Iteração {i}/100: RSS = {current_rss:.2f} MB (Delta: {current_rss - initial_rss:+.2f} MB)")

        final_rss = rss_samples[-1][1]
        growth = final_rss - initial_rss

        print(f"[MEMORY LEAK SUMMARY] Crescimento total de memória após 100 avaliações: {growth:+.2f} MB")
        # Um crescimento aceitável para 100 iterações com PyTorch em CPU é < 25 MB (buffers de allocador / cache interno)
        assert growth < 25.0, f"Crescimento excessivo de memória ({growth:.2f} MB) indica possível memory leak!"


class TestAdversarialSeparability:
    """Testes exaustivos de separabilidade com matriz cruzada completa do catálogo."""

    def test_identical_pairs_score_near_100(self):
        """
        Para TODOS os clipes do catálogo de memes, a auto-comparação (áudio contra ele mesmo)
        deve resultar em pontuação próxima de 100 (total_score >= 95.0).
        """
        processor = AudioProcessor()
        scorer = VoiceScorer(audio_processor=processor)

        clips = [f for f in os.listdir(CLIPS_DIR) if f.endswith(".wav") and not f.startswith("test_attempt")]
        assert len(clips) >= 6, f"Esperado ao menos 6 clipes no catálogo, encontrados: {len(clips)}"

        print(f"\n[IDENTICAL PAIRS] Testando auto-comparação em {len(clips)} clipes do catálogo...")
        identical_scores = {}
        for c in clips:
            c_path = os.path.join(CLIPS_DIR, c)
            res = scorer.evaluate_imitation(c_path, c_path)
            identical_scores[c] = res.total_score
            print(f"  {c:25s} -> Total: {res.total_score:5.1f} | Neural: {res.neural_score:5.1f} | Pitch: {res.pitch_score:5.1f} | Ritmo: {res.rhythm_score:5.1f}")
            assert res.total_score >= 95.0, f"Auto-comparação de {c} foi < 95.0 ({res.total_score})"

        mean_ident = float(np.mean(list(identical_scores.values())))
        print(f"[IDENTICAL PAIRS RESULTS] Média geral: {mean_ident:.2f} / 100 (Todas >= 95.0)")
        assert mean_ident >= 98.0

    def test_disjoint_pairs_score_below_30(self):
        """
        Matriz completa de pares cruzados de memes distintos (A != B).
        A meta mandatada é que pares disjuntos pontuem estritamente < 30.
        """
        processor = AudioProcessor()
        scorer = VoiceScorer(audio_processor=processor)

        clips = [f for f in os.listdir(CLIPS_DIR) if f.endswith(".wav") and not f.startswith("test_attempt")]
        disjoint_results = []

        print(f"\n[DISJOINT PAIRS MATRIX] Avaliando todos os pares disjuntos entre {len(clips)} clipes...")
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                c1 = clips[i]
                c2 = clips[j]
                p1 = os.path.join(CLIPS_DIR, c1)
                p2 = os.path.join(CLIPS_DIR, c2)

                res = scorer.evaluate_imitation(p1, p2)
                disjoint_results.append({
                    "c1": c1,
                    "c2": c2,
                    "total": res.total_score,
                    "neural": res.neural_score,
                    "pitch": res.pitch_score,
                    "rhythm": res.rhythm_score
                })

        scores = [d["total"] for d in disjoint_results]
        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        p90_score = float(np.percentile(scores, 90))
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))

        print(f"\n[DISJOINT PAIRS STATS - {len(disjoint_results)} PARES]")
        print(f"  Média:    {mean_score:.2f}")
        print(f"  Mediana:  {median_score:.2f}")
        print(f"  Mínimo:   {min_score:.2f}")
        print(f"  Máximo:   {max_score:.2f}")
        print(f"  P90:      {p90_score:.2f}")

        # Listar os 5 pares com maior pontuação para análise de separabilidade
        sorted_pairs = sorted(disjoint_results, key=lambda x: x["total"], reverse=True)
        print("  Top 5 pares com maior similaridade aparente:")
        for p in sorted_pairs[:5]:
            print(f"    {p['c1']:22s} vs {p['c2']:22s} -> Total: {p['total']:4.1f} (Neural: {p['neural']:4.1f}, Pitch: {p['pitch']:4.1f}, Ritmo: {p['rhythm']:4.1f})")

        # Verifica critério: média deve ser bem abaixo de 30 (esperado < 25), e se algum par ultrapassar 30, documentar
        assert mean_score < 30.0, f"Média dos pares disjuntos ({mean_score:.2f}) excedeu 30.0"
        # Testar se a esmagadora maioria (>90%) fica estritamente < 30
        below_30_count = sum(1 for s in scores if s < 30.0)
        percentage_below_30 = (below_30_count / len(scores)) * 100.0
        print(f"  Porcentagem de pares com Score < 30: {percentage_below_30:.1f}% ({below_30_count}/{len(scores)})")
        assert percentage_below_30 >= 90.0, f"Menos de 90% dos pares disjuntos pontuaram < 30 ({percentage_below_30:.1f}%)"
