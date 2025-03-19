import whisper  # Importa a biblioteca Whisper para reconhecimento de fala.
from idiomas_mapeamento import obter_nome_completo_idioma  # Importa a função para obter o nome completo do idioma.

def transcribe_audio(audio_path, model_name="base", detect_language_every_chunk=False):
    """
    Transcreve um arquivo de áudio, detectando a linguagem.

    Args:
        audio_path (str): Caminho para o arquivo de áudio.
        model_name (str): Nome do modelo Whisper.
        detect_language_every_chunk (bool): Detectar a lingua em cada chunk.

    Returns:
        tuple: Linguagem detectada e texto transcrito.
    """
    model = whisper.load_model(model_name)  # Carrega o modelo Whisper especificado.
    audio = whisper.load_audio(audio_path)  # Carrega o arquivo de áudio.

    # Verifica se o áudio é menor ou igual ao tamanho de um chunk (30 segundos).
    if len(audio) <= whisper.audio.N_SAMPLES:
        audio = whisper.pad_or_trim(audio)  # Preenche ou corta o áudio para ter exatamente 30 segundos.
        mel = whisper.log_mel_spectrogram(audio).to(model.device)  # Calcula o espectrograma log-Mel do áudio e o move para o dispositivo do modelo.
        _, probs = model.detect_language(mel)  # Detecta o idioma do áudio.
        codigo_linguagem = max(probs, key=probs.get)  # Obtém o código do idioma com a maior probabilidade.
        linguagem_completa = obter_nome_completo_idioma(codigo_linguagem)  # Obtém o nome completo do idioma.
        options = whisper.DecodingOptions()  # Cria um objeto com as opções de decodificação padrão.
        result = whisper.decode(model, mel, options)  # Decodifica o espectrograma log-Mel.
        return linguagem_completa, result.text # Retorna a linguagem e o texto.

    chunk_size = whisper.audio.N_SAMPLES  # Define o tamanho de cada bloco (chunk) como 30 segundos.
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]  # Divide o áudio em blocos de 30 segundos.

    transcription = ""  # Inicializa uma string vazia para armazenar a transcrição completa.
    linguagem_completa = ""  # Inicializa uma string vazia para armazenar o nome completo do idioma.
    # Itera sobre cada bloco (chunk), o i é o index do chunk.
    for i, chunk in enumerate(chunks):
        chunk = whisper.pad_or_trim(chunk)  # Preenche com silêncio ou corta o bloco para que tenha exatamente 30 segundos.
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)  # Calcula o espectrograma log-Mel do bloco e o move para o dispositivo do modelo.

        # Detectar a linguagem dependendo do parametro.
        # se for o primeiro chunk ou se o parametro for True, irá detectar a lingua.
        if not transcription or detect_language_every_chunk:
            _, probs = model.detect_language(mel)  # Detecta o idioma do áudio.
            codigo_linguagem = max(probs, key=probs.get)  # Obtém o código do idioma com a maior probabilidade.
            linguagem_completa = obter_nome_completo_idioma(codigo_linguagem)  # Obtém o nome completo do idioma.

        options = whisper.DecodingOptions()  # Cria um objeto com as opções de decodificação padrão.
        result = whisper.decode(model, mel, options)  # Decodifica o espectrograma log-Mel.
        transcription += result.text + " "  # Adiciona o texto transcrito do bloco à transcrição completa, com um espaço entre os blocos.

    return linguagem_completa, transcription.strip()  # Retorna o nome completo do idioma e a transcrição completa (removendo espaços extras no início e no fim).
