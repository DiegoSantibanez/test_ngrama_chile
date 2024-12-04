import streamlit as st
import pandas as pd
from collections import defaultdict, Counter
import random

class NgramTextPredictor:
    def __init__(self, n=3):
        """
        Inicializa un modelo de n-gramas para predicción de texto.
        
        Args:
            n (int): Tamaño del n-grama (por defecto 3 para trigramas)
        """
        self.n = n
        # Almacena conteos de n-gramas
        self.ngram_counts = defaultdict(Counter)
        # Almacena contextos únicos
        self.contexts = set()
        
    def train(self, corpus):
        """
        Entrena el modelo con un corpus de texto.
        
        Args:
            corpus (list): Lista de oraciones o documentos de texto
        """
        # Preprocesa el texto
        processed_corpus = [self._preprocess(texto) for texto in corpus]
        
        # Genera n-gramas y cuenta frecuencias
        for texto in processed_corpus:
            palabras = texto.split()
            
            # Asegura que haya suficientes palabras para crear n-gramas
            if len(palabras) < self.n:
                continue
            
            # Genera n-gramas
            for i in range(len(palabras) - self.n + 1):
                # Separa contexto (n-1 palabras) y palabra objetivo
                contexto = tuple(palabras[i:i+self.n-1])
                palabra_objetivo = palabras[i+self.n-1]
                
                # Incrementa conteo
                self.ngram_counts[contexto][palabra_objetivo] += 1
                self.contexts.add(contexto)
        
    def _preprocess(self, texto):
        """
        Preprocesa el texto (puede personalizar según necesidad)
        
        Args:
            texto (str): Texto a preprocesar
        
        Returns:
            str: Texto preprocesado
        """
        # Convierte a minúsculas y elimina signos de puntuación
        return texto.lower().strip()
    
    def predict(self, contexto, num_predicciones=3):
        """
        Predice las siguientes palabras dado un contexto.
        
        Args:
            contexto (tuple): Contexto de palabras previas
            num_predicciones (int): Número de predicciones a devolver
        
        Returns:
            list: Lista de predicciones ordenadas por probabilidad
        """
        # Ajusta el contexto al tamaño correcto
        contexto = tuple(contexto[-(self.n-1):])
        
        # Si el contexto no existe, elige un contexto aleatorio
        if contexto not in self.contexts:
            contexto = random.choice(list(self.contexts))
        
        # Obtiene conteos para el contexto
        conteos_contexto = self.ngram_counts[contexto]
        
        # Calcula probabilidades
        predicciones = sorted(
            conteos_contexto.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [pred[0] for pred in predicciones[:num_predicciones]]
    
    def generate_text(self, contexto_inicial, longitud=10):
        """
        Genera texto basado en predicciones de n-gramas.
        
        Args:
            contexto_inicial (list): Lista de palabras iniciales
            longitud (int): Longitud del texto a generar
        
        Returns:
            str: Texto generado
        """
        texto_generado = list(contexto_inicial)
        
        for _ in range(longitud):
            contexto_actual = texto_generado[-self.n+1:]
            predicciones = self.predict(contexto_actual)
            
            if not predicciones:
                break
            
            texto_generado.append(predicciones[0])
        
        return ' '.join(texto_generado)

def main():
    st.title("N-Gram Text Prediction Chat")
    
    # Cargar datos de entrenamiento
    try:
        df = pd.read_csv('texto_ngrama.csv')
        corpus = df['Texto del Comentario'].tolist()
    except FileNotFoundError:
        # Corpus de respaldo si no se encuentra el archivo
        corpus = [
            "me gusta programar en python",
            "python es un lenguaje increíble", 
            "me gusta aprender nuevas tecnologías",
            "la programación es divertida"
        ]
        st.warning("Archivo CSV no encontrado. Usando corpus de respaldo.")
    
    # Inicializar y entrenar el modelo
    predictor = NgramTextPredictor(n=3)
    predictor.train(corpus)
    
    # Inicializar historial de mensajes en la sesión
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input del usuario
    if prompt := st.chat_input("Escribe un mensaje..."):
        # Añadir mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta del modelo
        # Usa las últimas dos palabras como contexto
        palabras = prompt.lower().split()
        contexto = palabras[-2:] if len(palabras) >= 2 else palabras
        
        # Generar texto
        respuesta = predictor.generate_text(
            contexto_inicial=contexto, 
            longitud=5
        )
        
        # Añadir respuesta del modelo
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        
        # Mostrar respuesta del modelo
        with st.chat_message("assistant"):
            st.markdown(respuesta)

if __name__ == "__main__":
    main()