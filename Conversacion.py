import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
MAX_MENSAJES = 20

st.title("Mi Bot Informatico")

load_dotenv()
clave = st.secrets["GEMINI_API_KEY"]

history = StreamlitChatMessageHistory(key="chat_messages")


@st.cache_resource
def modelo():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=clave,
        temperature=0.7,
        max_output_tokens=5000
    )


@st.cache_resource
def modelo_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=clave
    )


llm = modelo()
embeddings = modelo_embeddings()

st.sidebar.title("Documentacion")
archivo_subido = st.sidebar.file_uploader("Sube tus pdfs aqui", type=["pdf"])

retriever = None

if archivo_subido:

    nombre_temporal = f"temp_{archivo_subido.name}"
    with open(nombre_temporal, "wb") as f:
        f.write(archivo_subido.getvalue())

    with st.sidebar.spinner("Indexando base de datos de conocimientos... "):

        cargador = PyPDFLoader(nombre_temporal)
        documentos = cargador.load()

        divisor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        fragmentos = divisor.split_documents(documentos)

        vector_store = Chroma.from_documents(fragmentos, embeddings)

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    st.sidebar.success(f"¡{archivo_subido.name} procesado con éxito!")

    os.remove(nombre_temporal)

instruccion = ("Eres un experto en informática. Responde con entusiasmo, "
               "menciona datos informáticos curiosos cada vez que puedas y "
               "a veces suelta expresiones divertidas. "
               "Importante: Si te proveen informacion bajo la seccion 'Contexto', usala prioritariamente"
               "para responder la duda del usuario. Si la informacion no esta ahí, usa tu conocimiento general."
               "Contexto: \n{contexto}"
               )

prompt = ChatPromptTemplate.from_messages([
    ("system", instruccion),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

for mensaje in history.messages:
    rol = "user" if mensaje.type == "human" else "assistant"

    with st.chat_message(rol):
        st.markdown(mensaje.content)

if mensaje_usuario := st.chat_input("Escribe tu mensaje"):
    with st.chat_message("user"):
        st.markdown(mensaje_usuario)

    texto_contexto = "No hay documentos adicionales cargados."
    if retriever:
        documentos_relacionados = retriever.invoke(mensaje_usuario)
        texto_contexto = "\n\n".join([doc.page_content for doc in documentos_relacionados])

    with st.chat_message("assistant"):
        with st.spinner("Procesando bits..."):
            respuesta = chain.invoke({
                "input": mensaje_usuario,
                "chat_history": history.messages[-MAX_MENSAJES:],
                "contexto": texto_contexto
            })

            if isinstance(respuesta.content, list):
                texto = "".join(
                    bloque.get("text", "")
                    for bloque in respuesta.content
                    if isinstance(bloque, dict)
                )
            else:
                texto = respuesta.content

            st.markdown(texto)

    history.add_message(
        HumanMessage(content=mensaje_usuario)
    )

    history.add_message(
        AIMessage(content=texto)
    )
