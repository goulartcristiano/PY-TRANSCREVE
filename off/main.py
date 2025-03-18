# main.py
from diarizacao import diarize_audio, format_diarization_output
from transcricao_full import transcribe_audio
import whisper
import datetime
import numpy as np

def transcribe_with_diarization(audio_file):
    """
    Transcreve um arquivo de áudio com o Whisper e adiciona a diarização de falantes em blocos (chunks).

    Args:
        audio_file (str): O caminho para o arquivo de áudio.

    Returns:
        str: A saída combinada da transcrição e da diarização.
    """
    # Obtém a linguagem e a transcrição completa usando a função transcribe_audio
    linguagem, full_transcription = transcribe_audio(audio_file, detect_language_every_chunk=False)

    # Exibe a linguagem detectada
    print(f"Linguagem detectada no arquivo: {linguagem}")

    # Carrega o modelo Whisper para obter os timestamps das palavras.
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_file)
    chunk_size = whisper.audio.N_SAMPLES  # 30 segundos (tamanho padrão dos blocos do Whisper)
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)] # Divide o áudio em blocos de 30 segundos

    diarization = diarize_audio(audio_file) # Realiza a diarização do áudio, identificando quem fala quando.

    output = "" # Inicializa uma string vazia para armazenar a saída formatada.

    for chunk in chunks: # Itera sobre cada bloco de áudio.
        options = whisper.DecodingOptions() # Define as opções de decodificação do Whisper.
        #result = model.transcribe(chunk, **options.__dict__) # aqui está a correção
        # Transcreve o bloco atual de áudio usando o modelo Whisper.
        # **options.__dict__ passa as opções de decodificação como argumentos nomeados.
        # O método transcribe do whisper espera um array numpy, mas chunk é um array numpy, então não precisa de alteração
        result = model.transcribe(chunk, **options.__dict__)

        for segment in result["segments"]:  # Itera sobre cada segmento de fala detectado pelo Whisper dentro do bloco.
            segment_start_time = datetime.timedelta(seconds=segment["start"]) # Converte o tempo de início do segmento para um objeto timedelta.
            segment_end_time = datetime.timedelta(seconds=segment["end"]) # Converte o tempo de fim do segmento para um objeto timedelta.
            formatted_time = str(segment_start_time).split(".")[0] # Formata o tempo de início para o formato HH:MM:SS.
            text = segment["text"] # Obtém o texto transcrito do segmento.

            speaker = "" # Inicializa uma string vazia para armazenar o nome do falante.
            for turn, _, spk in diarization.itertracks(yield_label=True): # Itera sobre cada turno de fala identificado pela diarização.
                # Verifica se o segmento está dentro do turno atual.
                if turn.start <= segment["start"] <= turn.end:
                    speaker = spk # Atribui o nome do falante ao segmento.
                    break # Sai do loop de turnos, pois já encontramos o falante do segmento.

            if speaker: # Se um falante foi identificado para o segmento.
                output += f"{formatted_time} - {speaker}: {text}\n" # Adiciona o texto, o falante e o tempo à saída formatada.

    return linguagem, output # Retorna a linguagem e a saída formatada com a transcrição e a diarização.

# Exemplo de Uso
audio_file = "audio_curto.mp3"  # Substitua pelo seu arquivo de áudio.

# Chama a função principal
linguagem_detectada, final_output = transcribe_with_diarization(audio_file)

# Salva a saída em um arquivo
output_filename = "transcricao_com_diarizacao.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(f"Linguagem Detectada: {linguagem_detectada}\n") # Escreve a linguagem detectada no início do arquivo.
    f.write(final_output) # Escreve o texto da transcrição e diarização no arquivo

print(f"Transcrição combinada com diarização salva em: {output_filename}")
