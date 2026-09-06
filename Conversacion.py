import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

st.title("Mi bot informatico")


load_dotenv()



clave = st.secrets["GEMINI_API_KEY"]


@st.cache_resource
def obtener_cliente(api_key):
    
    return genai.Client(api_key=api_key)


if not clave:
    st.error("🔑 Falta la API Key de Gemini. Configúrala en los Secrets de Streamlit.")
    st.stop()

cliente = obtener_cliente(clave)

if "historial" not in st.session_state:
    st.session_state.historial = []

instruccion = ("Eres un experto en informatica. Responde con entusiasmo, menciona datos informaticos "
               "cada vez que puedas y a veces suelta expresiones divertidas.")

# Dibujar el historial en pantalla
for mensaje in st.session_state.historial:
    rol = "user" if mensaje["role"] == "user" else "assistant"
    with st.chat_message(rol):
        st.markdown(mensaje["parts"][0]["text"])

if mensaje_usuario := st.chat_input("Escribe tu mensaje"):
    with st.chat_message("user"):
        st.markdown(mensaje_usuario)

    st.session_state.historial.append({
        "role": "user",
        "parts": [{"text": mensaje_usuario}]
    })

    with st.chat_message("assistant"):
        with st.spinner("Procesando bits..."):
            try:
                
                respuesta = cliente.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=st.session_state.historial,
                    config=types.GenerateContentConfig(
                        system_instruction=instruccion,
                        temperature=0.7,
                        max_output_tokens=1000
                    )
                )
                st.markdown(respuesta.text)
                
                # Guardamos la respuesta exitosa en el historial
                st.session_state.historial.append({
                    "role": "model",
                    "parts": [{"text": respuesta.text}]
                })
            except Exception as e:
                st.error(f"Error al generar contenido: {e}")
