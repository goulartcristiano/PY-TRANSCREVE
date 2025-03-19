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
}

def obter_nome_completo_idioma(codigo_idioma):
    return LINGUAGENS.get(codigo_idioma, codigo_idioma)
