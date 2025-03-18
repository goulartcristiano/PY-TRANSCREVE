LINGUAGENS = {
    "en": "Inglês",
    "pt": "Português",
    "es": "Espanhol",
    "fr": "Francês",
    "de": "Alemão",
    "it": "Italiano",
    "ja": "Japonês",
    "zh": "Chinês",
    "ko": "Coreano",
    "ru": "Russo",
    # Adicione mais idiomas conforme necessário
}

def obter_nome_completo_idioma(codigo_idioma):
    """Retorna o nome completo do idioma com base no código de duas letras.

    Args:
        codigo_idioma (str): O código de idioma de duas letras (ex: "pt", "en").

    Returns:
        str: O nome completo do idioma (ex: "Português", "Inglês") ou o próprio código se não for encontrado.
    """
    return LINGUAGENS.get(codigo_idioma, codigo_idioma)
