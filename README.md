### README.md

# **Gerenciamento de Histórico e ChatBot Baseado em LangChain e Groq**

Este repositório contém um exemplo de implementação de um ChatBot utilizando o framework **LangChain** integrado com **Groq**. O programa demonstra como gerenciar históricos de mensagens, criar templates dinâmicos para prompts e otimizar interações com o modelo de linguagem.

---

## **Índice**
1. [Pré-requisitos](#pré-requisitos)
2. [Instalação](#instalação)
3. [Descrição dos Blocos de Código](#descrição-dos-blocos-de-código)
4. [Executando o Programa](#executando-o-programa)
5. [Contribuição](#contribuição)
6. [Licença](#licença)

---

## **Pré-requisitos**
- Python 3.8 ou superior.
- Conta na plataforma **Groq** e uma chave de API válida.
- Arquivo `.env` com a variável de ambiente `GROQ_API_KEY` configurada.

---

## **Instalação**

1. Clone este repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
   cd SEU_REPOSITORIO
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure sua chave de API no arquivo `.env`:
   ```
   GROQ_API_KEY=YOUR_GROQ_API_KEY
   ```

---

## **Descrição dos Blocos de Código**

### **Importação das Bibliotecas**

```python
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
```

Essas bibliotecas são necessárias para:
- **`dotenv`**: Carregar variáveis de ambiente.
- **`langchain_*`**: Criar e gerenciar históricos de mensagens, templates de prompts e integração com modelos de linguagem.
- **`operator`**: Facilitar a manipulação de dicionários.

---

### **Carregando Variáveis de Ambiente**

```python
load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

Carrega a chave da API do Groq armazenada no arquivo `.env`.

---

### **Inicialização do Modelo**

```python
model = ChatGroq(model_name="Gemma2-9b-it", api_key=GROQ_API_KEY)
```

Inicializa o modelo de linguagem com o nome especificado e a chave de API.

---

### **Exemplo 1: Gerenciamento de Histórico de Mensagens**

#### **Criação e Recuperação de Históricos**

```python
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

Este bloco gerencia múltiplos históricos de mensagens para sessões distintas, garantindo contexto em interações contínuas.

#### **Ligação entre Histórico e Modelo**

```python
with_message_history = RunnableWithMessageHistory(model, get_session_history)
config = {"configurable": {"session_id": "chat1"}}

response = with_message_history.invoke(
    [HumanMessage(content="Oi, meu nome é Silmara e eu sou economista")],
    config=config
)
```

Conecta o modelo ao histórico e processa mensagens do usuário.

---

### **Exemplo 2: Uso de Prompt Templates**

#### **Criação do Template**

```python
prompt = ChatPromptTemplate.from_messages(
    [("system", "Você é um assistente útil. Responda todas as perguntas com precisão no idioma."),
     MessagesPlaceholder(variable_name="messages")]
)
```

Permite criar uma entrada estruturada para o modelo com mensagens dinâmicas.

#### **Conexão com o Modelo**

```python
chain = prompt | model
response = chain.invoke({"messages": [HumanMessage(content="Oi, meu nome é Silmara")]})
```

Aplica o template ao modelo para interagir com o usuário.

---

### **Gerenciamento de Memória do ChatBot**

```python
trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)
```

O `trimmer` limita o número de mensagens armazenadas para evitar exceder o contexto máximo do modelo.

---

### **Criação de um Pipeline de Execução**

```python
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {"messages": messages + [HumanMessage(content="Qual sorvete eu gosto")]}
)
```

Combina múltiplos componentes (otimização do histórico, prompt e modelo) em um fluxo contínuo de execução.

---

## **Executando o Programa**

1. Certifique-se de que o ambiente virtual está ativado.
2. Execute o programa principal:
   ```bash
   python main.py
   ```

---

## **Contribuição**

Contribuições são bem-vindas! Sinta-se à vontade para abrir **issues** ou enviar **pull requests**.

---

## **Licença**

Este projeto está licenciado sob a [MIT License](LICENSE).

--- 

Este README detalha cada bloco do código, facilitando sua compreensão e execução. Se precisar de ajustes ou melhorias, sinta-se à vontade para contribuir!