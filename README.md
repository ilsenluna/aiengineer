# Master AI Engineer · LangChain . Ilsen Luna

Proyecto de clase: las piezas basicas de LangChain, una por archivo, en el orden de la sesion.
Todo corre en local con Ollama (llama3.1): coste de API 0 EUR.

## Requirements

Ollama installed and model llama3.1 downloaded

## Setup
1. Create the environment: `python -m venv .venv` 
2. Activate `.venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Verify Ollama: http://localhost:11434/ 

## Homework

**Summarizer with LCEL**: `prompt | model | parser` chain that summarizes a long text into 3 bullet points.

## Challenge

**Chat with memory**: a chat with memory that uses at least one tool during a multi-turn conversation.