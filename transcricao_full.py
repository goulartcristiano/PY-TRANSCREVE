# transcricao_full.py
import whisper  # Importa a biblioteca Whisper para reconhecimento de fala.
import numpy as np  # Importa a biblioteca NumPy para operações numéricas (usada internamente pelo Whisper).
from idiomas_mapeamento import obter_nome_completo_idioma  # Importa a função para obter o nome completo do idioma.

def transcribe_audio(audio_path, model_name="base", detect_language_every_chunk=False):
    """
    Transcreve um arquivo de áudio em blocos (chunks) e detecta a linguagem.

    Args:
        audio_path (str): O caminho para o arquivo de áudio.
        model_name (str): O nome do modelo whisper a ser usado (o padrão é "base").
        detect_language_every_chunk (bool): se for True, detecta a lingua em cada chunk, se for false detecta apenas no primeiro.

    Returns:
        tuple: Uma tupla contendo a linguagem detectada (str) e o texto transcrito (str).
    """
    model = whisper.load_model(model_name)  # Carrega o modelo Whisper especificado.
    audio = whisper.load_audio(audio_path)  # Carrega o arquivo de áudio.

    chunk_size = whisper.audio.N_SAMPLES  # Define o tamanho de cada bloco (chunk) como 30 segundos.
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]  # Divide o áudio em blocos de 30 segundos.

    transcription = ""  # Inicializa uma string vazia para armazenar a transcrição completa.
    linguagem_completa = ""  # Inicializa uma string vazia para armazenar o nome completo do idioma.

    for i, chunk in enumerate(chunks):  # Itera sobre cada bloco (chunk), o i é o index do chunk.
        chunk = whisper.pad_or_trim(chunk)  # Preenche com silêncio ou corta o bloco para que tenha exatamente 30 segundos.
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)  # Calcula o espectrograma log-Mel do bloco e o move para o dispositivo do modelo.

        # Detectar a linguagem dependendo do parametro.
        if not transcription or detect_language_every_chunk: # se for o primeiro chunk ou se o parametro for True, irá detectar a lingua.
            _, probs = model.detect_language(mel)  # Detecta o idioma do áudio.
            codigo_linguagem = max(probs, key=probs.get)  # Obtém o código do idioma com a maior probabilidade.
            linguagem_completa = obter_nome_completo_idioma(codigo_linguagem)  # Obtém o nome completo do idioma.
            # print(f"Linguagem detectada: {linguagem}")

        options = whisper.DecodingOptions()  # Cria um objeto com as opções de decodificação padrão.
        result = whisper.decode(model, mel, options)  # Decodifica o espectrograma log-Mel.
        transcription += result.text + " "  # Adiciona o texto transcrito do bloco à transcrição completa, com um espaço entre os blocos.

    return linguagem_completa, transcription.strip()  # Retorna o nome completo do idioma e a transcrição completa (removendo espaços extras no início e no fim).
