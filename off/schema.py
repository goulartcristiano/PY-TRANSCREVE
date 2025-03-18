# instalar openai-whisper
# instalar ffmpeg: https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

#VERSAO BASICA
"""import whisper

modelo = whisper.load_model("base")

resposta = modelo.transcribe("Gravando2.m4a")

print(resposta)"""

#VERSAO NORMAL
"""import whisper

model = whisper.load_model("base")

# carrega o áudio 
audio = whisper.load_audio("Gravando2.m4a")
audio = whisper.pad_or_trim(audio) # preenche/corta para caber em 30 segundos

# cria espectrograma log-Mel e move para o mesmo dispositivo que o modelo
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detecta a linguagem falada
_, probs = model.detect_language(mel)
print(f"Linguagem detectada: {max(probs, key=probs.get)}")

# decodifica o áudio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# imprime o texto reconhecido
print(result.text)"""


#VERSAO PARA AUDIO +30seg CRIADO PELO GEMINI
import whisper  # Importa a biblioteca whisper para reconhecimento de fala
import numpy as np  # Importa a biblioteca numpy, que é usada internamente pelo whisper para manipulação de arrays

def transcribe_audio(audio_path, model_name="base"):
    """Transcreve um arquivo de áudio em blocos (chunks).

    Args:
        audio_path (str): O caminho para o arquivo de áudio.
        model_name (str): O nome do modelo whisper a ser usado (o padrão é "base").
    Returns:
        str: O texto transcrito do áudio.
    """
    model = whisper.load_model(model_name)  # Carrega o modelo whisper especificado (ex: "base", "small", "medium", "large")
    audio = whisper.load_audio(audio_path)  # Carrega o arquivo de áudio usando a função load_audio do whisper

    # Se o áudio for mais curto do que 30 segundos (N_SAMPLES), podemos processar normalmente
    if len(audio) <= whisper.audio.N_SAMPLES:
        audio = whisper.pad_or_trim(audio)  # Ajusta o tamanho do áudio para 30 segundos
        mel = whisper.log_mel_spectrogram(audio).to(model.device)  # Calcula o espectrograma log-Mel do áudio e o move para o dispositivo do modelo (CPU ou GPU)
        _, probs = model.detect_language(mel)  # Detecta o idioma do áudio
        print(f"Linguagem detectada: {max(probs, key=probs.get)}")  # Imprime o idioma detectado com maior probabilidade
        options = whisper.DecodingOptions()  # Cria um objeto com as opções de decodificação padrão
        result = whisper.decode(model, mel, options)  # Decodifica o espectrograma log-Mel usando o modelo e as opções
        return result.text  # Retorna o texto transcrito

    # Se o áudio for maior do que 30 segundos, vamos processar em blocos (chunks)
    chunk_size = whisper.audio.N_SAMPLES  # Define o tamanho de cada bloco (chunk) como 30 segundos (em samples de áudio)
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]  # Divide o áudio em blocos de 30 segundos

    transcription = ""  # Inicializa uma string vazia para armazenar a transcrição completa
    for chunk in chunks:  # Itera sobre cada bloco (chunk)
        # pad_or_trim para que o chunk tenha o tamanho esperado (30 segundos)
        chunk = whisper.pad_or_trim(chunk)  # Ajusta o tamanho do bloco atual para 30 segundos (preenche com silêncio ou corta se necessário)
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)  # Calcula o espectrograma log-Mel do bloco e o move para o dispositivo do modelo

        # Detectar a linguagem só uma vez, no primeiro bloco
        if not transcription:
            _, probs = model.detect_language(mel)  # Detecta o idioma do áudio (apenas no primeiro bloco)
            print(f"Linguagem detectada: {max(probs, key=probs.get)}")  # Imprime o idioma detectado com maior probabilidade

        options = whisper.DecodingOptions()  # Cria um objeto com as opções de decodificação padrão
        result = whisper.decode(model, mel, options)  # Decodifica o espectrograma log-Mel usando o modelo e as opções
        transcription += result.text + " "  # Concatena o texto transcrito do bloco atual ao resultado total, adicionando um espaço

    return transcription.strip()  # Retorna a transcrição completa, removendo espaços extras no início e no fim

# Exemplo de uso
transcribed_text = transcribe_audio("Gravando.m4a")  # Chama a função para transcrever o arquivo "Gravando2.m4a"

# Salva em arquivo txt
output_filename = "transcricao.txt" # Define o nome do arquivo de saída
with open(output_filename, "w", encoding="utf-8") as output_file: # Abre o arquivo em modo de escrita (w)
    output_file.write(transcribed_text) # Escreve o texto transcrito no arquivo
print(f"Transcrição salva em: {output_filename}") # Imprime uma mensagem informando que o arquivo foi salvo e onde