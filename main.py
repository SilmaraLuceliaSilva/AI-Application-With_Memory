# Importação das Bibliotecas Necessários
import os
from dotenv import load_dotenv, find_dotenv                                                 # Biblioteca para carregar variáveis de ambiente
from langchain_groq import ChatGroq                                                         # Integração do LangChain com Groq
from langchain_community.chat_message_histories import ChatMessageHistory                   #  Histórico de mensagens
from langchain_core.chat_history import BaseChatMessageHistory                              # Classe base para histórico
from langchain_core.runnables.history import RunnableWithMessageHistory                     # Permite gerenciar histórico dinamicamente
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder                  # Criação de templates para prompts
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages   # Manipulação de mensagens
from langchain_core.runnables import RunnablePassthrough                                    # Para criar fluxos de execução reutilizáveis
from operator import itemgetter                                                             # Facilita a extração de valores de dicionários

# Carrega as variáveis de ambiente do arquivo .env (para proteger credenciais)

load_dotenv(find_dotenv())

# Obtém a chave da API do Groq armazenada nas variáveis de ambiente
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializa o modelo de IA utilizando a API do Groq
model = ChatGroq(model_name= "Gemma2-9b-it", api_key=GROQ_API_KEY)

# Exemplo 1 -----------------------------------------------------
# Dicionário para armazenar históricos de conversação por sessão
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Recupera ou cria um histórico de mensagens para uma determinada sessão.
    Isso permite manter um contexto contínuo para diferentes usuários ou interações.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory ()
    return store[session_id]

# Cria um gerenciador de histórico que conecta o modelo ao armazenamento de mensagens
with_message_history= RunnableWithMessageHistory(model, get_session_history)

# Configuração da sessão (identificador único para cada chat)
config = {"configurable":{"session_id": "chat1"} }

# Exemplo de interação inicial do usuário
response = with_message_history.invoke (
    [HumanMessage(content="Oi, meu nome é Silmara e eu sou economista")],
    config=config
)

# Exemplo 21 ----------------------------------------------------------

# Criação de um prompt template para estruturar a entrada do modelo
prompt = ChatPromptTemplate.from_messages(
     [("system", "Você é um assistente útil. Responda todas as perguntas com precisão no idioma ."), MessagesPlaceholder(variable_name="messages")  # Permite adicionar mensagens dinamicamente
    ])

# Conecta o modelo ao template de prompt
chain = prompt | model

# Exemplo de interação usando o template
response = chain.invoke({"messages": [HumanMessage(content="Oi, meu nome é Silmara")]})


# Gerenciamento da memória do chatbot
trimmer = trim_messages( 
    max_tokens=45, # Define um limite máximo de tokens para evitar ultrapassar o contexto
    strategy="last", # Mantém as últimas mensagens mais recentes
    token_counter=model, # Usa o modelo para contar os tokens
    include_system=True, # Inclui a mensagem do sistema no histórico
    allow_partial=False, # Evita que mensagens fiquem cortadas
    start_on="human" # Começa a contagem com mensagens humanas
)

# Exemplo de histórico de mensagens
messages = [
    SystemMessage(content="Você é um bom assistente. Responda todas as perguntas com precisão do idioma."),
    HumanMessage(content="Oi! Meu nome é João"),
    AIMessage(content="Oi, João! Como posso te ajudar?"),
    HumanMessage(content="Eu gosto de sorvete de limão"),
]

# Aplica o limitador de memória ao histórico de mensagens
response = trimmer.invoke(messages)

# Exibe a resposta inicial do chatbot
#print("Mensagens após trimmer:", response)

# Criando um pipeline de execução para otimizar a passagem de informações
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)  # Aplica a otimização do histórico
    | prompt  # Passa a entrada pelo template de prompt
    | model  # Envia para o modelo
)

# Exemplo de interação utilizando o pipeline otimizado
response = chain.invoke(
    {"messages": messages +[HumanMessage(content="Qual sorvete eu gosto")],
    
    }
)

# Exibe a resposta final do modelo
print("Resposta final:", response.content)