from diarizacao import diarize_audio  # Importa a função para diarização de áudio.
from transcricao import transcribe_audio  # Importa a função para transcrição de áudio.
import whisper  # Importa a biblioteca Whisper para reconhecimento de fala.
import datetime  # Importa o módulo datetime para manipulação de datas e horas.

def transcribe_with_diarization(audio_file):
    """
    Transcreve um arquivo de áudio com Whisper e adiciona a diarização.

    Args:
        audio_file (str): Caminho para o arquivo de áudio.

    Returns:
        tuple: Uma tupla contendo a linguagem detectada (str) e a saída combinada da transcrição e da diarização (str).
    """
    # Chama a função transcribe_audio para obter a linguagem detectada e a transcrição completa.
    # detect_language_every_chunk=False indica que a detecção de idioma será feita apenas uma vez no início.
    linguagem, full_transcription = transcribe_audio(audio_file, detect_language_every_chunk=False)
    print(f"Linguagem detectada no arquivo: {linguagem}")  # Imprime a linguagem detectada.

    model = whisper.load_model("base")  # Carrega o modelo Whisper "base".
    audio = whisper.load_audio(audio_file)  # Carrega o arquivo de áudio.
    options = whisper.DecodingOptions()  # Define as opções de decodificação padrão.
    result = model.transcribe(audio, **options.__dict__)  # Transcreve o áudio inteiro usando o modelo Whisper.

    diarization = diarize_audio(audio_file)  # Chama a função diarize_audio para obter a diarização do áudio.
    output = ""  # Inicializa uma string vazia para armazenar a saída formatada.

    # Itera sobre cada segmento na transcrição do Whisper.
    for segment in result["segments"]:
        segment_start_time = datetime.timedelta(seconds=segment["start"])  # Converte o tempo de início do segmento para um objeto timedelta.
        segment_end_time = datetime.timedelta(seconds=segment["end"])  # Converte o tempo de fim do segmento para um objeto timedelta.
        formatted_time = str(segment_start_time).split(".")[0]  # Formata o tempo de início para o formato HH:MM:SS.
        text = segment["text"]  # Obtém o texto transcrito do segmento.

        speaker = ""  # Inicializa uma string vazia para armazenar o falante do segmento.
        # Itera sobre cada turno na diarização.
        for turn, _, spk in diarization.itertracks(yield_label=True):
            # Verifica se o tempo de início do segmento está dentro do turno.
            if turn.start <= segment["start"] <= turn.end:
                speaker = spk  # Define o falante do segmento.
                break  # Sai do loop interno, pois já encontrou o falante.

        # Se um falante foi encontrado para o segmento.
        if speaker:
            output += f"{formatted_time} - {speaker}: {text}\n"  # Adiciona a linha formatada à saída.

    return linguagem, output  # Retorna a linguagem detectada e a saída formatada.

# Exemplo de Uso
audio_file = "audio_curto.mp3"  # Define o nome do arquivo de áudio.

linguagem_detectada, final_output = transcribe_with_diarization(audio_file)  # Chama a função principal para transcrever e diarizar o áudio.

output_filename = "transcricao_com_diarizacao.txt"  # Define o nome do arquivo de saída.
# Abre o arquivo de saída em modo de escrita com codificação UTF-8.
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(f"Linguagem Detectada: {linguagem_detectada}\n")  # Escreve a linguagem detectada no arquivo.
    f.write(final_output)  # Escreve a saída formatada no arquivo.

print(f"Transcrição combinada com diarização salva em: {output_filename}")  # Imprime uma mensagem informando que o arquivo foi salvo.