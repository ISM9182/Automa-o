%%writefile agents.py
# agents.py

from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI

from tools import (
    ColabVideoToAudioTool,
    ColabWhisperTranscriptionTool,
    FileManagerTool,
)

class MediaProcessingAgents:
    def __init__(self):
        # Certifique-se de que OPENAI_API_KEY está configurada nos secrets do Colab
        # O modelo gpt-4o é recomendado para melhores resultados
        self.gpt_model = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    def media_processor_agent(self):
        return Agent(
            role=dedent("""Especialista em manipulação e conversão de formatos de mídia."""),
            backstory=dedent("""\
                Você é um engenheiro de mídia com vasta experiência em processamento de vídeo e áudio.
                Sua expertise inclui o uso de bibliotecas avançadas para converter, extrair e otimizar
                arquivos de mídia, garantindo a melhor qualidade e compatibilidade para as próximas etapas
                do fluxo de trabalho."""),
            goal=dedent("""\
                Converter arquivos de vídeo em áudio MP3 de alta qualidade e gerenciar eficientemente
                as operações de arquivo no ambiente do Google Colab (Google Drive montado)."""),
            verbose=True, # Define se o agente mostrará pensamentos e ações detalhadas
            allow_delegation=False, # Define se o agente pode delegar tarefas para outros agentes
            llm=self.gpt_model, # O modelo de linguagem grande (LLM) que o agente usará
            tools=[ # Ferramentas que o agente tem acesso para executar suas tarefas
                ColabVideoToAudioTool(),
                FileManagerTool()
            ]
        )

    def audio_transcriber_agent(self):
        return Agent(
            role=dedent("""Linguista computacional e especialista em reconhecimento de fala."""),
            backstory=dedent("""\
                Você é um perito em transformar fala em texto, com um profundo conhecimento de modelos
                de reconhecimento automático de fala (ASR), como o Whisper. Sua missão é garantir
                transcrições precisas e contextualmente corretas a partir de arquivos de áudio,
                superando desafios como ruídos e sotaques."""),
            goal=dedent("""\
                Transcrever arquivos de áudio MP3 em transcrições textuais precisas e completas,
                otimizando o uso do modelo Whisper para os melhores resultados."""),
            verbose=True,
            allow_delegation=False,
            llm=self.gpt_model,
            tools=[
                ColabWhisperTranscriptionTool(),
                FileManagerTool()
            ]
        )

    def didactic_content_creator_agent(self):
        return Agent(
            role=dedent("""Criador de Conteúdo Didático e Formatador HTML."""),
            backstory=dedent("""\
                Você é um editor de conteúdo experiente, com uma paixão por transformar informações
                complexas em materiais educacionais claros, concisos e envolventes. Sua especialidade
                inclui a organização de conteúdo para fácil compreensão e a formatação em HTML,
                garantindo que o material seja não apenas informativo, mas também visualmente acessível."""),
            goal=dedent("""\
                Criar resumos didáticos e explicativos a partir de transcrições textuais e formatá-los
                em HTML acessível para fins educacionais."""),
            verbose=True,
            allow_delegation=False,
            llm=self.gpt_model,
            tools=[
                FileManagerTool(), # Este agente só precisa da ferramenta de gerenciamento de arquivos para ler a transcrição e escrever o HTML
            ]
        )
