import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

st.title("Mi bot informatico")

load_dotenv()
clave = os.getenv("GEMINI_API_KEY")


@st.cache_resource
def obtener_cliente():
    return genai.Client(api_key=clave)


cliente = obtener_cliente()

if "historial" not in st.session_state:
    st.session_state.historial = []

instruccion = "Eres un expero en informatica. Responde con entusiasmo, menciona datos informaticos cada vez que " \
              "puedas y a veces suelta expresiones divertidas."

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
            respuesta = cliente.models.generate_content(
                model="gemini-3.6-flash",
                contents=st.session_state.historial,
                config=types.GenerateContentConfig(system_instruction=instruccion,
                                                   temperature=0.7,
                                                   max_output_tokens=1000)

            )
            st.markdown(respuesta.text)

    st.session_state.historial.append({
        "role": "model",
        "parts": [{"text": respuesta.text}]
    })
