from pyannote.audio import Pipeline
import torch
import datetime
import pyannote.core

# Carrega o pipeline de diarização de falantes.
# Criar um token de acesso do HuggingFace salvo no ambiente. Verificar com: huggingface-cli whoami
try:
    # Tenta carregar o pipeline pré-treinado para diarização de falantes (versão 3.1).
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    # Descomentar a próxima linha para usar a GPU.
    # pipeline.to(torch.device("cuda"))
except Exception as e:
    # Se ocorrer um erro ao carregar o pipeline, imprime uma mensagem de erro.
    print(f"Erro ao carregar o pipeline de diarização: {e}")
    print("Por favor, certifique-se de que você executou 'huggingface-cli login' no seu terminal e que seu token é válido.")
    # Encerra o programa se o pipeline não puder ser carregado.
    exit()

def diarize_audio(audio_file):
    """
    Realiza a diarização de falantes no arquivo de áudio fornecido.

    Args:
        audio_file (str): Caminho para o arquivo de áudio.

    Returns:
        pyannote.core.Annotation: A saída da diarização.
    """
    try:
        # Tenta executar o pipeline de diarização no arquivo de áudio.
        diarization = pipeline(audio_file)
        # Retorna o resultado da diarização.
        return diarization
    except Exception as e:
        # Se ocorrer um erro durante a diarização, imprime uma mensagem de erro.
        print(f"Erro durante a diarização: {e}")
        # Retorna None em caso de erro.
        return None

def format_diarization_output(diarization, audio_file):
    """
    Formata a saída da diarização em uma string mais legível.

    Args:
        diarization (pyannote.core.Annotation): A saída da diarização.
        audio_file(str): O nome do arquivo de áudio
    Returns:
        str: A saída da diarização formatada.
    """
    # Se a diarização for None (houve um erro), retorna uma string vazia.
    if diarization is None:
        return ""
    
    # Inicia a string de saída com informações sobre o arquivo de áudio.
    output_str = f"Saída da diarização para: {audio_file}\n\n"
    # Itera sobre cada turno de fala na diarização.
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Converte o tempo de início do turno para um objeto timedelta.
        timestamp = datetime.timedelta(seconds=turn.start)
        # Formata o tempo para o formato HH:MM:SS.
        formatted_time = str(timestamp).split(".")[0]
        # Adiciona uma linha à string de saída com o tempo e o nome do falante.
        output_str += f"{formatted_time} : {speaker}\n"
    # Retorna a string de saída formatada.
    return output_str
