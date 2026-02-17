import streamlit as st
import os
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# 1. Configuración de la Página - Estilo Trend Hunter
st.set_page_config(page_title="TREND ALERT | No Bullshit News", page_icon="🔥", layout="wide")
st.title("🔥 TREND ALERT")
st.markdown("### Lo que es tendencia ahora mismo, sin filtros ni rollos.")

# 2. Sidebar: Configuración de API Keys
with st.sidebar:
    st.header("🔑 ACCESO")
    google_key = st.text_input("Google API Key:", type="password")
    tavily_key = st.text_input("Tavily API Key:", type="password")
    
    if google_key and tavily_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        st.success("✅ Estás dentro. Sistema listo.")

# 3. Definición del Estado y el Grafo (Tu lógica intacta)
class AgentState(TypedDict):
    question: str
    search_results: str
    final_story: str

def tool_search_news(state: AgentState):
    """Busca en tiempo real usando Tavily"""
    search = TavilySearchResults(max_results=3)
    results = search.invoke(state["question"])
    return {"search_results": str(results)}

def generator_story(state: AgentState):
    """Transforma las noticias en un breakdown súper cool"""
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash') 
    
    prompt = f"""
    Eres un 'Trend Spotter' experto y muy carismático. Tu estilo es parecido al de un creador de contenido de éxito en redes sociales (estilo TikTok/Twitter pero escrito).
    
    Tu misión: Explicar lo que está pasando de forma súper directa, con lenguaje actual, usando anglicismos comunes (hype, red flag, plot twist, vibe, mood) si encajan, y muchos emojis.
    
    CONTEXTO DE LAS NOTICIAS:
    {state['search_results']}
    
    TEMA QUE ME TIENES QUE CONTAR:
    {state['question']}
    
    INSTRUCCIÓN: No me cuentes un cuento. Hazme un 'breakdown' rápido.
    - Empieza con un titular con mucho gancho.
    - Cuéntame el 'té' (la info importante) de forma clara pero con estilo.
    - Dime por qué esto es importante o por qué hay tanto 'hype'.
    - Sé divertido y un poco irreverente.
    """
    
    response = llm.invoke(prompt)
    return {"final_story": response.content}

# Construcción del flujo (Tal cual lo tenías)
workflow = StateGraph(AgentState)
workflow.add_node("buscador", tool_search_news)
workflow.add_node("escritor", generator_story)

workflow.set_entry_point("buscador")
workflow.add_edge("buscador", "escritor")
workflow.add_edge("escritor", END)

app_graph = workflow.compile()

# 4. Interfaz de Usuario (Input con estilo)
if google_key and tavily_key:
    pregunta = st.text_input("🎤 ¿De qué hay que hablar hoy?", 
                             placeholder="Ej: ¿Qué está pasando con el nuevo modelo de OpenAI?")

    if pregunta:
        with st.spinner("🚀 Rastreando la red..."):
            try:
                # Ejecución del grafo
                inputs = {"question": pregunta}
                resultado = app_graph.invoke(inputs)
                
                # Resultado principal
                st.markdown("---")
                st.markdown(f"## ⚡ El Breakdown:")
                st.write(resultado["final_story"])
                
                # Datos técnicos por si acaso
                with st.expander("🛠️ Ver los 'receipts' (Fuentes de Tavily)"):
                    st.code(resultado["search_results"], language="text")
            
            except Exception as e:
                st.error(f"Se cayó la conexión: {str(e)}")

else:
    st.warning("👈 Pon las claves en el menú para activar el radar de tendencias.")
