from pyannote.audio import Pipeline

try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
except Exception as e:
    print(f"Erro ao carregar o pipeline de diarização: {e}")
    print("Por favor, certifique-se de que você executou 'huggingface-cli login' no seu terminal e que seu token é válido.")
    exit()

def diarize_audio(audio_file):
    """
    Realiza a diarização de falantes no arquivo de áudio.

    Args:
        audio_file (str): Caminho para o arquivo de áudio.

    Returns:
        pyannote.core.Annotation: A saída da diarização.
    """
    try:
        diarization = pipeline(audio_file)
        return diarization
    except Exception as e:
        print(f"Erro durante a diarização: {e}")
        return None
