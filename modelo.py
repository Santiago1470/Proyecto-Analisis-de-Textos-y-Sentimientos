import geopandas as gpd
import pandas as pd
import json
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    set_seed
)
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_turismo = None
vectorizer = None
vectores_lugares = None
generador = None
ultimas_respuestas = []
max_historial = 5
contexto_conversacion = []

stop_words_es = [
        'de', 'la', 'en', 'el', 'y', 'a', 'que', 'es', 'se', 'del', 'las', 'los',
        'un', 'una', 'con', 'por', 'para', 'al', 'lo', 'como', 'más', 'o', 'pero'
]

vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stop_words_es,
        max_features=2000,
        min_df=1,
        max_df=0.9
)


df_turismo = gpd.read_file('turismo.geojson')

if 'LATITUD' in df_turismo.columns:
  df_turismo['latitud'] = df_turismo['LATITUD']
if 'LONGITUD' in df_turismo.columns:
  df_turismo['longitud'] = df_turismo['LONGITUD']

columnas_eliminar = ['geometry', 'OBJECTID', 'CODIGO', 'LATITUD', 'LONGITUD', 'ID', 'ICONOGRAFI']
df_turismo = df_turismo.drop(columns=[col for col in columnas_eliminar if col in df_turismo.columns])

for col in ['CORREO_PRO', 'TELEFONO', 'NOMBRE_PRO', 'DIRECCION']:
  if col in df_turismo.columns:
    df_turismo[col] = df_turismo[col].fillna('')

df_turismo['TIPO_DE_PA'] = df_turismo['TIPO_DE_PA'].fillna('lugar turístico')

df_turismo['texto_busqueda'] = (
  df_turismo['NOMBRE'].fillna('') + ' ' +
  df_turismo['TIPO_DE_PA'].fillna('') + ' ' +
  df_turismo['DIRECCION'].fillna('')
).str.lower()

textos_validos = [texto for texto in df_turismo['texto_busqueda'] if len(texto.strip()) > 3]
if textos_validos:
    vectores_lugares = vectorizer.fit_transform(textos_validos)


respuestas_naturales = {
    'museo': {
        'intros': [
            "Perfecto! Para cultura y arte te recomiendo",
            "Si te gusta el arte, definitivamente debes visitar",
            "Para una experiencia cultural increíble, ve a",
            "Los museos son geniales aquí. Te sugiero",
            "Excelente elección! Un museo que te va a encantar es"
        ],
        'descripciones': [
            "Es un lugar fascinante donde puedes sumergirte en la historia y el arte",
            "Tiene exposiciones increíbles que realmente valen la pena",
            "Es perfecto para pasar unas horas aprendiendo y disfrutando del arte",
            "Te va a sorprender la calidad de sus colecciones",
            "Es uno de esos lugares que te dejan con ganas de volver"
        ],
        'consejos': [
            "Tip: Los domingos muchos museos tienen entrada gratuita",
            "Consejo: Lleva cédula por si hay descuento de estudiante",
            "Recomendación: Ve en la mañana, hay menos gente",
            "Dato: Algunos tienen cafeterías muy buenas adentro"
        ]
    },
    'parque': {
        'intros': [
            "Qué buena idea relajarte! Te recomiendo muchísimo",
            "Para desconectarte un rato, nada mejor que",
            "Si buscas tranquilidad y naturaleza, ve a",
            "Para relajarte al aire libre, te sugiero",
            "Perfecto plan! Un parque hermoso es"
        ],
        'descripciones': [
            "Es ideal para caminar, relajarte o simplemente sentarte a leer un libro",
            "Tiene zonas verdes preciosas donde puedes respirar aire fresco",
            "Es perfecto para desestresarte y conectar con la naturaleza",
            "Puedes hacer ejercicio, meditar o simplemente contemplar el paisaje",
            "Es uno de esos lugares donde realmente puedes desconectarte del ruido de la ciudad"
        ],
        'consejos': [
            "Tip: Los fines de semana hay más actividades",
            "Consejo: Lleva algo para beber, caminar da sed",
            "Recomendación: Las mañanas son más frescas y tranquilas",
            "Dato: Algunos tienen zonas de picnic muy chéveres"
        ]
    },
    'iglesia': {
        'intros': [
            "Para encontrar paz y admirar arquitectura hermosa, te recomiendo",
            "Si buscas tranquilidad espiritual, visita",
            "Una iglesia realmente impresionante es",
            "Para un momento de reflexión y belleza arquitectónica, ve a",
            "Te va a encantar la historia y arquitectura de"
        ],
        'descripciones': [
            "Su arquitectura es impresionante y transmite una paz increíble",
            "Es un lugar perfecto para la reflexión y la contemplación",
            "Tiene detalles arquitectónicos que realmente te van a sorprender",
            "Es un oasis de tranquilidad en medio de la ciudad",
            "La historia que guarda entre sus muros es fascinante"
        ],
        'consejos': [
            "Tip: Respeta los horarios de misa si vas a visitarla",
            "Consejo: Las fotos del exterior son espectaculares",
            "Recomendación: Ve en horas de la tarde, la luz es hermosa",
            "Dato: Muchas tienen historias fascinantes que puedes preguntar"
        ]
    },
    'centro_comercial': {
        'intros': [
            "Si buscas shopping y entretenimiento, definitivamente",
            "Para compras y diversión, te recomiendo",
            "Un centro comercial súper completo es",
            "Si quieres pasar el día comprando y comiendo rico, ve a",
            "Para plan de shopping y cine, perfecto"
        ],
        'descripciones': [
            "Tiene de todo: tiendas, restaurantes, cine y muchas opciones de entretenimiento",
            "Es perfecto para pasar el día completo entre compras y comida",
            "Tiene una variedad increíble de tiendas y lugares para comer",
            "Es uno de esos centros comerciales donde puedes encontrar lo que necesites",
            "Perfecto para días lluviosos o cuando quieres un plan bajo techo"
        ]
    }
}
respuestas_situacionales = {
    'cerca_de': [
        "Ahí cerca tienes",
        "Muy cerca de esa zona está",
        "A pocas cuadras encuentras",
        "En esa misma área tienes",
        "No muy lejos de ahí está"
    ],
    'zona_especifica': [
        "En esa zona te recomiendo",
        "Por ahí tienes",
        "En ese sector está",
        "Justo en esa área puedes ir a",
        "Perfecto, en esa zona está"
    ],
    'primera_vez': [
        "Si es tu primera vez por acá, te recomiendo",
        "Para empezar a conocer Bogotá, perfecto",
        "Como introducción a la ciudad, ve a",
        "Para que te enamores de Bogotá, visita"
    ]
}

info_bogota = {
    'transporte': {
        'transmilenio': "El TransMilenio te cuesta $2,950 y llega prácticamente a todos lados",
        'uber': "También puedes usar Uber, que es súper cómodo",
        'caminar': "Si no está muy lejos, caminar por Bogotá puede ser muy agradable",
        'apps': "Te recomiendo usar Google Maps o Moovit para las rutas exactas"
    },
    'precios': {
        'museos': "Los museos normalmente cuestan entre $10,000 y $25,000",
        'parques': "Lo mejor es que la mayoría de parques son completamente gratis",
        'iglesias': "Las iglesias no cobran entrada, son abiertas al público",
        'descuentos': "Muchos lugares tienen descuentos para estudiantes y tercera edad",
        'domingos': "Los domingos varios museos son gratis - aprovecha!"
    },
    'horarios': {
        'general': "La mayoría abren de 9 AM a 5 PM",
        'museos': "Ojo que muchos museos cierran los lunes",
        'fines_semana': "Los fines de semana a veces tienen horarios diferentes"
    }
}


model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

generador = pipeline(
  "text-generation",
  model=model_name,
  tokenizer=tokenizer,
  device=0 if device.type == 'cuda' else -1,
  max_length=100,
  do_sample=True,
  temperature=0.7,
  pad_token_id=tokenizer.eos_token_id
)


def clasificar_intencion(consulta):
    consulta_lower = consulta.lower()

    intenciones = []

    if any(palabra in consulta_lower for palabra in ['cómo llegar', 'como llego', 'llegar', 'ir a', 'transporte', 'como ir', 'cómo voy']):
        intenciones.append('transporte')
    if any(palabra in consulta_lower for palabra in ['precio', 'cuesta', 'cobran', 'entrada', 'gratis', 'cuanto', 'vale']):
        intenciones.append('precio')
    if any(palabra in consulta_lower for palabra in ['horario', 'abierto', 'cerrado', 'hora', 'cuando', 'qué hora']):
        intenciones.append('horario')
    if any(palabra in consulta_lower for palabra in ['recomiendas', 'mejor', 'buenos', 'qué hacer', 'visitar', 'conocer', 'quiero', 'puedo ir']):
        intenciones.append('recomendacion')
    if any(palabra in consulta_lower for palabra in ['cerca', 'zona', 'sur', 'norte', 'centro', 'por', 'en']):
        intenciones.append('ubicacion')
    if any(palabra in consulta_lower for palabra in ['relajar', 'descansar', 'tranquilo', 'paz', 'calma']):
        intenciones.append('relajacion')

    return intenciones if intenciones else ['general']


def detectar_tipo_lugar(consulta, lugar_info=None):
    consulta_lower = consulta.lower()

    if lugar_info:
        tipo_lugar = lugar_info.get('tipo', '').lower()
        nombre_lugar = lugar_info.get('nombre', '').lower()
        texto_completo = f"{tipo_lugar} {nombre_lugar}"
    else:
        texto_completo = consulta_lower

    tipos_detectados = []

    if any(word in texto_completo for word in ['museo', 'galeria', 'arte', 'exposicion', 'cultural']):
        tipos_detectados.append('museo')
    if any(word in texto_completo for word in ['iglesia', 'templo', 'catedral', 'basilica', 'religiosa']):
        tipos_detectados.append('iglesia')
    if any(word in texto_completo for word in ['parque', 'verde', 'natural', 'recreativo', 'jardin']):
        tipos_detectados.append('parque')
    if any(word in texto_completo for word in ['biblioteca', 'libros', 'lectura']):
        tipos_detectados.append('biblioteca')
    if any(word in texto_completo for word in ['teatro', 'espectaculo', 'obra']):
        tipos_detectados.append('teatro')
    if any(word in texto_completo for word in ['centro comercial', 'mall', 'shopping', 'compras']):
        tipos_detectados.append('centro_comercial')
    if any(word in texto_completo for word in ['plaza', 'plazoleta', 'plazuela']):
        tipos_detectados.append('plaza')

    return tipos_detectados[0] if tipos_detectados else 'general'



def buscar_lugares(consulta, max_resultados=3):
    if vectores_lugares is None or len(consulta.strip()) < 2:
        return []

    consulta_lower = consulta.lower()
    expansion_palabras = {
        'museo': ['museo', 'galeria', 'arte', 'cultura', 'exposicion', 'artistico'],
        'iglesia': ['iglesia', 'templo', 'catedral', 'basilica', 'religiosa', 'sagrada'],
        'parque': ['parque', 'verde', 'natural', 'recreativo', 'deportivo', 'jardin', 'bosque'],
        'relajar': ['parque', 'verde', 'tranquilo', 'paz', 'naturaleza', 'descanso'],
        'sur': ['sur', 'sureño', 'meridional', 'south'],
        'centro': ['centro', 'céntrico', 'central', 'downtown'],
        'norte': ['norte', 'norteño', 'septentrional', 'north'],
        'carrera': ['carrera', 'calle', 'avenida', 'via'],
        'comercial': ['centro comercial', 'mall', 'shopping', 'tiendas', 'compras']
    }

    consulta_expandida = consulta_lower
    for categoria, palabras in expansion_palabras.items():
        if categoria in consulta_lower:
            consulta_expandida += " " + " ".join(palabras)

    consulta_vector = vectorizer.transform([consulta_expandida])
    similitudes = cosine_similarity(consulta_vector, vectores_lugares).flatten()
    nombres_usados_recientes = {resp.get('nombre_usado', '') for resp in ultimas_respuestas[-2:]}
    lugares_candidatos = []
    for idx, (_, lugar) in enumerate(df_turismo.iterrows()):
        if idx >= len(similitudes):
            continue

        nombre_lugar = str(lugar['NOMBRE'])
        if nombre_lugar in nombres_usados_recientes:
            continue

        if similitudes[idx] > 0.05:
            lugares_candidatos.append({
                'nombre': nombre_lugar,
                'tipo': str(lugar['TIPO_DE_PA']),
                'direccion': str(lugar['DIRECCION']) if pd.notna(lugar['DIRECCION']) else '',
                'telefono': str(lugar['TELEFONO']) if pd.notna(lugar['TELEFONO']) else '',
                'administrador': str(lugar['NOMBRE_PRO']) if pd.notna(lugar['NOMBRE_PRO']) else '',
                'latitud': lugar.get('latitud', None),
                'longitud': lugar.get('longitud', None),
                'correo': str(lugar["CORREO_PRO"]) if pd.notna(lugar["CORREO_PRO"]) else '',
                'puntuacion': similitudes[idx]
            })

    lugares_candidatos.sort(key=lambda x: x['puntuacion'], reverse=True)
    return lugares_candidatos[:max_resultados]


def respuesta_sin_lugares(consulta, intenciones):
    consulta_lower = consulta.lower()

    respuestas_naturales_sin_lugares = {
        'recomendacion': [
            "Claro! Bogotá tiene lugares increíbles. Te tira más la cultura (museos, teatros), la naturaleza (parques chéveres), la historia (iglesias antiguas) o el shopping (centros comerciales)? Con un poquito más de detalle te puedo recomendar algo perfecto",
            "Por supuesto! Qué onda te llama más? Algo cultural, natural, histórico o más comercial? Dependiendo de tu mood te puedo sugerir lugares geniales",
            "Súper! Bogotá tiene de todo. Estás más para museos y arte, parques y naturaleza, iglesias e historia o centros comerciales? Dime qué te late más y te doy recomendaciones bacanas"
        ],

        'transporte': [
            f"Para moverte por Bogotá tienes varias opciones: {info_bogota['transporte']['transmilenio']}, {info_bogota['transporte']['uber']} o {info_bogota['transporte']['caminar']}. Hay algún lugar específico al que quieres llegar?",
            f"Perfecto! {info_bogota['transporte']['transmilenio']} y llega a casi todos lados. {info_bogota['transporte']['apps']} A dónde específicamente quieres ir?"
        ],

        'precio': [
            f"Los precios varían: {info_bogota['precios']['museos']}, pero {info_bogota['precios']['parques']}. {info_bogota['precios']['domingos']} Qué tipo de plan tienes en mente?",
            f"Te cuento: {info_bogota['precios']['museos']}, {info_bogota['precios']['parques']} y {info_bogota['precios']['iglesias']}. Algo específico que te interese?"
        ],

        'ubicacion': [
            "Excelente pregunta! Bogotá está llena de opciones geniales. El centro tiene museos históricos súper bacanos, el norte centros comerciales modernos y restaurantes top, y el sur parques naturales hermosos. En qué zona prefieres buscar?",
            "Cada zona de Bogotá tiene su encanto: centro para historia y cultura, norte para modernidad y shopping, sur para naturaleza. Cuál te llama más?"
        ],

        'relajacion': [
            "Qué plan tan bueno! Para relajarte Bogotá tiene parques preciosos. Prefieres algo más natural y verde, o algo más urbano pero tranquilo? También hay lugares como bibliotecas que son súper pacíficos",
            "Perfecto para desconectarse! Te provoca más un parque con naturaleza, una zona verde para caminar, o tal vez algo más cultural pero relajante como una biblioteca bonita?"
        ],

        'general': [
            "Hola! Soy tu asistente turístico de Bogotá. Puedo recomendarte museos geniales, parques para relajarte, iglesias históricas, centros comerciales y mucho más. Qué tipo de plan te provoca hoy?",
            "Qué chimba que quieras conocer Bogotá! Tengo recomendaciones de cultura, naturaleza, historia, entretenimiento y más. Qué te llama la atención?",
            "Bienvenido a Bogotá! Te puedo ayudar con museos, parques, iglesias, shopping y todos los planes bacanos de la ciudad. Por dónde empezamos?"
        ]
    }

    if 'recomendacion' in intenciones:
        return random.choice(respuestas_naturales_sin_lugares['recomendacion'])
    elif 'transporte' in intenciones:
        return random.choice(respuestas_naturales_sin_lugares['transporte'])
    elif 'precio' in intenciones:
        return random.choice(respuestas_naturales_sin_lugares['precio'])
    elif 'ubicacion' in intenciones:
        return random.choice(respuestas_naturales_sin_lugares['ubicacion'])
    elif 'relajacion' in intenciones:
        return random.choice(respuestas_naturales_sin_lugares['relajacion'])
    else:
        return random.choice(respuestas_naturales_sin_lugares['general'])
    

def generar_respuesta_natural(consulta, lugares, intenciones):
    if not lugares:
        return respuesta_sin_lugares(consulta, intenciones)

    lugar = lugares[0]
    tipo_detectado = detectar_tipo_lugar(consulta, lugar)
    consulta_lower = consulta.lower()
    es_ubicacion_especifica = any(palabra in consulta_lower for palabra in ['cerca', 'por', 'en', 'zona'])
    respuestas_tipo = respuestas_naturales.get(tipo_detectado, respuestas_naturales.get('museo'))
    if es_ubicacion_especifica:
        intro = random.choice(respuestas_situacionales['zona_especifica'])
    else:
        intro = random.choice(respuestas_tipo['intros'])

    respuesta = f"{intro} {lugar['nombre']}"

    if lugar['direccion'] and lugar['direccion'] != 'nan' and len(lugar['direccion']) > 5:
        if 'carrera' in consulta_lower or 'calle' in consulta_lower:
            respuesta += f" (está en {lugar['direccion']})"
        else:
            respuesta += f", que queda en {lugar['direccion']}"

    respuesta += f". {random.choice(respuestas_tipo['descripciones'])}"

    if 'transporte' in intenciones:
        opciones_transporte = [
            f"Para llegar: {info_bogota['transporte']['transmilenio']}",
            f"{info_bogota['transporte']['uber']}",
            f"{info_bogota['transporte']['apps']}"
        ]
        respuesta += f"\n\n{random.choice(opciones_transporte)}."

    elif 'precio' in intenciones:
        if tipo_detectado == 'museo':
            respuesta += f"\n\nSobre precios: {info_bogota['precios']['museos']}. {info_bogota['precios']['domingos']}"
        elif tipo_detectado == 'parque':
            respuesta += f"\n\nLo mejor: {info_bogota['precios']['parques']}"
        elif tipo_detectado == 'iglesia':
            respuesta += f"\n\nGenial: {info_bogota['precios']['iglesias']}"
        else:
            respuesta += f"\n\nPrecios: Te recomiendo preguntar directamente para tener la info más actualizada."

    elif 'horario' in intenciones:
        respuesta += f"\n\nHorarios: {info_bogota['horarios']['general']}."
        if tipo_detectado == 'museo':
            respuesta += f" {info_bogota['horarios']['museos']}."

    if tipo_detectado in respuestas_naturales:
        consejos = respuestas_naturales[tipo_detectado].get('consejos', [])
        if consejos and random.random() > 0.4:
            respuesta += f"\n\n{random.choice(consejos)}"

    if lugar['telefono'] and len(lugar['telefono']) >= 7 and lugar['telefono'] != 'nan':
        respuesta += f"\n\nSi necesitas más info: {lugar['telefono']}{' o ' + lugar['correo'] if lugar['correo'] else ''}"

    return respuesta


def guardar_contexto(consulta, respuesta, lugar_nombre=None):
    global contexto_conversacion
    contexto_conversacion.append({
        'pregunta': consulta,
        'respuesta': respuesta,
        'lugar_usado': lugar_nombre,
        'timestamp': len(contexto_conversacion)
    })

    if len(contexto_conversacion) > 5:
        contexto_conversacion.pop(0)



def responder(consulta):
    global ultimas_respuestas

    if len(consulta.strip()) < 2:
        return "Hola! Pregúntame algo más específico sobre lugares turísticos en Bogotá y te ayudo con mucho gusto"

    intenciones = clasificar_intencion(consulta)
    lugares_encontrados = buscar_lugares(consulta)
    respuesta = generar_respuesta_natural(consulta, lugares_encontrados, intenciones)
    lugar_nombre = lugares_encontrados[0]['nombre'] if lugares_encontrados else None
    guardar_contexto(consulta, respuesta, lugar_nombre)
    if lugares_encontrados:
        ultimas_respuestas.append({
            'pregunta': consulta,
            'respuesta': respuesta,
            'nombre_usado': lugares_encontrados[0]['nombre']
        })

        if len(ultimas_respuestas) > max_historial:
            ultimas_respuestas.pop(0)

    return respuesta, lugares_encontrados