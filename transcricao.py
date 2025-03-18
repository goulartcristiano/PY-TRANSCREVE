# transcricao.py
import whisper  # Importa a biblioteca Whisper para reconhecimento de fala.
import numpy as np  # Importa a biblioteca NumPy para operações numéricas (usada internamente pelo Whisper).
from idiomas_mapeamento import obter_nome_completo_idioma  # Importa a função para obter o nome completo do idioma.

def transcribe_audio(audio_path, model_name="base"):
    """Transcreve um arquivo de áudio em blocos (chunks).

    Args:
        audio_path (str): O caminho para o arquivo de áudio.
        model_name (str): O nome do modelo whisper a ser usado (o padrão é "base").
    Returns:
        tuple: Uma tupla contendo o nome completo do idioma (str) e o texto transcrito (str).
    """
    model = whisper.load_model(model_name)  # Carrega o modelo Whisper especificado (ex: "base", "small", "medium", "large").
    audio = whisper.load_audio(audio_path)  # Carrega o arquivo de áudio usando a função load_audio do Whisper.

    # Verifica se o áudio é mais curto que 30 segundos (tamanho padrão do Whisper).
    if len(audio) <= whisper.audio.N_SAMPLES:
        audio = whisper.pad_or_trim(audio)  # Preenche com silêncio ou corta o áudio para que tenha exatamente 30 segundos.
        mel = whisper.log_mel_spectrogram(audio).to(model.device)  # Calcula o espectrograma log-Mel (representação visual do áudio) e o move para o dispositivo do modelo (CPU ou GPU).
        _, probs = model.detect_language(mel)  # Detecta o idioma do áudio, retornando um dicionário com probabilidades para cada idioma.
        codigo_linguagem = max(probs, key=probs.get)  # Obtém o código do idioma (ex: "pt", "en") com a maior probabilidade.
        linguagem_completa = obter_nome_completo_idioma(codigo_linguagem)  # Usa a função para obter o nome completo do idioma (ex: "Português").
        print(f"Linguagem detectada: {linguagem_completa}")  # Imprime o nome completo do idioma detectado.
        options = whisper.DecodingOptions()  # Cria um objeto com as opções de decodificação padrão do Whisper.
        result = whisper.decode(model, mel, options)  # Decodifica o espectrograma log-Mel usando o modelo e as opções, gerando a transcrição.
        return linguagem_completa, result.text  # Retorna o nome completo do idioma e o texto transcrito.

    # Se o áudio for maior que 30 segundos, ele será processado em blocos (chunks).
    chunk_size = whisper.audio.N_SAMPLES  # Define o tamanho de cada bloco como 30 segundos.
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]  # Divide o áudio em blocos de 30 segundos.

    transcription = ""  # Inicializa uma string vazia para armazenar a transcrição completa.
    linguagem_completa = ""  # Inicializa uma string vazia para armazenar o nome completo do idioma.
    for chunk in chunks:  # Itera sobre cada bloco (chunk).
        chunk = whisper.pad_or_trim(chunk)  # Preenche com silêncio ou corta o bloco para que tenha exatamente 30 segundos.
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)  # Calcula o espectrograma log-Mel do bloco e o move para o dispositivo do modelo.

        # Detecta o idioma apenas uma vez, no primeiro bloco.
        if not transcription:
            _, probs = model.detect_language(mel)  # Detecta o idioma do áudio no primeiro bloco.
            codigo_linguagem = max(probs, key=probs.get)  # Obtém o código do idioma com a maior probabilidade.
            linguagem_completa = obter_nome_completo_idioma(codigo_linguagem)  # Obtém o nome completo do idioma.
            print(f"Linguagem detectada: {linguagem_completa}")  # Imprime o nome completo do idioma.

        options = whisper.DecodingOptions()  # Cria um objeto com as opções de decodificação padrão.
        result = whisper.decode(model, mel, options)  # Decodifica o espectrograma log-Mel do bloco.
        transcription += result.text + " "  # Adiciona o texto transcrito do bloco à transcrição completa, com um espaço entre os blocos.

    return linguagem_completa, transcription.strip()  # Retorna o nome completo do idioma e a transcrição completa (removendo espaços extras no início e no fim).
