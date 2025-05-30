from fastapi import FastAPI, Query
# from analizar import analizar_sentimiento_hf
from modelo import responder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from textoInput import TextoInput
import json
import random

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analizarTexto")
def emotionsText_detect(texto_input: TextoInput):
    texto = {
  "type": "FeatureCollection",
  "name": "ITur",
  "crs": {
    "type": "name",
    "properties": {
      "name": "urn:ogc:def:crs:EPSG::3857"
    }
  },
  "features": [
    {
      "type": "Feature",
      "properties": {
        "OBJECTID_1": 1,
        "ID": 3499,
        "Codigo": "11001.1.1.2.2.6.1.3499",
        "Nombre": "Parque Cedro Golf Club",
        "Direccion": "Carrera 7A # 150 - 85",
        "Tipo_de_Pa": "Patrimonio cultural material inmueble",
        "Iconografi": "Atractivo Cultural",
        "Nombre_Pro": "Instituto Distrital de Recreación y Deporte - IDRD",
        "Direccio_1": "Calle 63 # 59A - 06",
        "Correo_Pro": "atncliente@idrd.gov.co",
        "Telefono": 5716605400,
        "Latitud": 4.725616,
        "Longitud": -74.026808
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-8240626.5716, 526650.600000002]
      }
    }]
    }

    texto = texto_input.texto

    resultado = responder(texto)
    print(resultado)

    lugar_nombre = ""
    if len(resultado[1]) != 0:
        lugar_nombre = resultado[1][0]["nombre"]
    # print(lugar_nombre)

    # Buscar el lugar en el archivo GeoJSON
    with open("turismo.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    lugar_encontrado = None
    for feature in geojson_data["features"]:
        props = feature["properties"]
        if props["NOMBRE"].strip().lower() == lugar_nombre.strip().lower():
            lugar_encontrado = {
                "nombre": props["NOMBRE"],
                "direccion": props.get("Direccion", ""),
                "tipo": props.get("Tipo_de_Pa", ""),
                "entidad": props.get("Nombre_Pro", ""),
                "correo": props.get("Correo_Pro", ""),
                "telefono": props.get("Telefono", ""),
                "lat": props["LATITUD"],
                "lng": props["LONGITUD"]
            }
            break

    if lugar_encontrado:
        return {
            "lugar": lugar_encontrado
        }, resultado[0]
    else:
        return {
            "lugar": None,
            "mensaje": f"No se encontró el lugar '{lugar_nombre}' en el geojson."
        }, resultado[0]

@app.get("/comentarios")
def buscar_comentarios(tipoComentario: str = Query(..., description="Tipo de comentario: positivo, neutral o negativo")):
    with open('comentarios.json', 'r', encoding='utf-8') as file:
        datos = json.load(file)
    comentarios = datos['comentarios']
    # print(comentarios)
    emociones = []
    for c in comentarios:
        resultado = analizar_sentimiento_hf(c['comentario'])
        
        if (stars[resultado[1]] == 'Positivo' or stars[resultado[1]] == 'Muy Positivo') and (tipoComentario == 'positivo'):
            print(resultado)
            emociones.append(c)
        elif stars[resultado[1]] == 'Neutro' and (tipoComentario == 'neutral'):
            emociones.append(c)
        elif (stars[resultado[1]] == 'Negativo' or stars[resultado[1]] == 'Muy Negativo') and (tipoComentario == 'negativo'):
            emociones.append(c)
    
    return {"Comentario": emociones}
