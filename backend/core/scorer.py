from dataclasses import dataclass
from typing import Optional
from rapidfuzz import fuzz

from backend.core.audio_processor import AudioProcessor

@dataclass
class ScoreResult:
    total_score: float
    neural_score: float
    pitch_score: float
    rhythm_score: float
    word_score: Optional[float]
    feedback: str
    details: dict

class VoiceScorer:
    def __init__(self, audio_processor: Optional[AudioProcessor] = None):
        self.processor = audio_processor or AudioProcessor()

    def evaluate_imitation(
        self,
        ref_audio_path: str,
        user_audio_path: str,
        target_text: Optional[str] = None,
        user_text: Optional[str] = None
    ) -> ScoreResult:
        """
        Executa a avaliação completa comparando o áudio original com a tentativa do usuário.
        """
        # 1. Similaridade da Rede Neural Siamesa (CRNN)
        s_neural = self.processor.compute_neural_similarity(ref_audio_path, user_audio_path)

        # 2. Similaridade de Contorno Melódico / Pitch (DTW)
        s_pitch = self.processor.compute_pitch_contour_similarity(ref_audio_path, user_audio_path)

        # 3. Similaridade de Ritmo / Cadência (RMS Envelope)
        s_rhythm = self.processor.compute_rhythm_similarity(ref_audio_path, user_audio_path)

        # 4. Similaridade de Palavras (se fornecidas transcrições/legendas)
        s_words = None
        if target_text and user_text:
            s_words = round(float(fuzz.token_sort_ratio(target_text.lower(), user_text.lower())), 2)

        # Cálculo da Nota Ponderada Final
        if s_words is not None:
            # 40% Rede Siamesa, 30% Pitch, 15% Ritmo, 15% Palavras
            total = (0.40 * s_neural) + (0.30 * s_pitch) + (0.15 * s_rhythm) + (0.15 * s_words)
        else:
            # Sem texto: 50% Rede Siamesa, 35% Pitch, 15% Ritmo
            total = (0.50 * s_neural) + (0.35 * s_pitch) + (0.15 * s_rhythm)

        total = round(float(total), 1)

        # Feedback narrativo / explicativo gerado pela IA
        feedback = self._generate_feedback(total, s_neural, s_pitch, s_rhythm, s_words)

        return ScoreResult(
            total_score=total,
            neural_score=s_neural,
            pitch_score=s_pitch,
            rhythm_score=s_rhythm,
            word_score=s_words,
            feedback=feedback,
            details={
                "is_model_trained": self.processor.is_model_loaded,
                "weights_used": "40% Neural, 30% Pitch, 15% Ritmo, 15% Palavras" if s_words else "50% Neural, 35% Pitch, 15% Ritmo"
            }
        )

    def _generate_feedback(self, total: float, neural: float, pitch: float, rhythm: float, words: Optional[float]) -> str:
        comments = []
        if total >= 88:
            comments.append("🌟 Atuação Lendária! Você capturou a alma do áudio original!")
        elif total >= 75:
            comments.append("🔥 Excelente imitação! Muito perto do original.")
        elif total >= 55:
            comments.append("👍 Boa tentativa! Dá para reconhecer bem a cena.")
        else:
            comments.append("😅 Ficou bem diferente do original, mas valeu a coragem!")

        # Diagnósticos específicos
        if pitch < 50:
            comments.append("A entonação (subidas e descidas do tom) destoou um pouco.")
        elif pitch >= 80:
            comments.append("A curva de entonação vocal foi muito precisa!")

        if rhythm < 50:
            comments.append("O timing e as pausas da frase ficaram fora do compasso.")
        elif rhythm >= 80:
            comments.append("O ritmo e a cadência foram no ponto exato!")

        if words is not None and words < 60:
            comments.append("Algumas palavras foram pronunciadas de forma incompleta ou errada.")

        return " ".join(comments)
