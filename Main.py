%%writefile main.py
# main.py

import os
from google.colab import userdata
from crewai import Crew, Process
from textwrap import dedent

from agents import MediaProcessingAgents
from tasks import CustomTasks

# --- Configuração da API Key OpenAI ---
try:
    # Tenta carregar a chave da API do OpenAI dos secrets do Colab
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
    print("OPENAI_API_KEY carregada dos secrets do Colab.")
except Exception as e:
    print(f"Erro ao carregar OPENAI_API_KEY dos secrets do Colab: {e}")
    print("Por favor, verifique se a chave 'OPENAI_API_KEY' está configurada corretamente nos secrets.")

class MediaProcessingCrew:
    def __init__(self, video_input_path, audio_output_path, transcription_output_path, html_output_path):
        self.video_input_path = video_input_path
        self.audio_output_path = audio_output_path
        self.transcription_output_path = transcription_output_path
        self.html_output_path = html_output_path
        
        # Instancia os agentes e as tarefas
        self.agents = MediaProcessingAgents()
        self.tasks = CustomTasks()

    def run(self):
        # Cria as instâncias dos agentes
        media_processor = self.agents.media_processor_agent()
        audio_transcriber = self.agents.audio_transcriber_agent()
        content_creator = self.agents.didactic_content_creator_agent()

        # Define as tarefas para cada agente
        process_media_task = self.tasks.process_media_task(
            agent=media_processor, 
            video_input_path=self.video_input_path, 
            audio_output_path=self.audio_output_path
        )
        transcribe_audio_task = self.tasks.transcribe_audio_task(
            agent=audio_transcriber, 
            audio_input_path=self.audio_output_path,
            transcription_output_path=self.transcription_output_path
        )
        didactic_content_creation_task = self.tasks.didactic_content_creation_task(
            agent=content_creator, 
            transcription_input_path=self.transcription_output_path,
            html_output_path=self.html_output_path
        )

        # Configura a Crew (equipe) com os agentes e as tarefas
        crew = Crew(
            agents=[media_processor, audio_transcriber, content_creator],
            tasks=[process_media_task, transcribe_audio_task, didactic_content_creation_task],
            verbose=True, # Mostra os passos detalhados da execução
            process=Process.sequential # Define que as tarefas serão executadas em sequência
        )

        # Inicia a execução da Crew
        return crew.kickoff()

if __name__ == "__main__":
    print(dedent("""
        ##############################################################
        ## Bem-vindo à Orquestração de Processamento de Média CrewAI! ##
        ##############################################################
    """))

    print("Google Drive já deve estar montado. Se não, verifique a Célula 1.")

    print("\n--- Por favor, forneça os caminhos completos para os arquivos ---")
    # Pede ao usuário os caminhos de entrada e saída
    video_input_path = input("Caminho do arquivo de vídeo de entrada (ex: /content/drive/MyDrive/Videos/meu_video.mp4): ").strip()
    audio_output_path = input("Caminho para o arquivo de áudio de saída (ex: /content/drive/MyDrive/Audios/audio_extraido.mp3): ").strip()
    transcription_output_path = input("Caminho para o arquivo de transcrição de saída (ex: /content/drive/MyDrive/Transcricoes/transcricao.txt): ").strip()
    html_output_path = input("Caminho para o arquivo HTML didático de saída (ex: /content/drive/MyDrive/HTMLs/resumo_final.html): ").strip()

    # Cria uma instância da Crew e a executa
    media_processing_crew = MediaProcessingCrew(
        video_input_path, 
        audio_output_path, 
        transcription_output_path, 
        html_output_path
    )
    
    print("\nIniciando o processo da CrewAI...")
    try:
        result = media_processing_crew.run()
    except Exception as e:
        print(f"\n########################")
        print(f"## ERRO CRÍTICO NA EXECUÇÃO DA CREW: {e}")
        print(f"########################\n")
        print("Verifique os logs acima para mais detalhes (especialmente erros de API, ferramentas ou caminhos de arquivo).")
        exit()

    print("\n\n########################")
    print("## Resultados Finais do Processamento de Média:")
    print("########################\n")
    print(result)
    print(f"\nVerifique os arquivos gerados em:")
    print(f"- Áudio: {audio_output_path}")
    print(f"- Transcrição: {transcription_output_path}")
    print(f"- HTML Didático: {html_output_path}")

    # Salva um resumo da saída em um arquivo local no Colab
    output_summary_file = 'media_processing_summary.txt'
    with open(output_summary_file, 'w', encoding='utf-8') as file:
        file.write("## Resumo do Processamento de Mídia da CrewAI:\n")
        file.write("########################\n\n")
        file.write(result)
        file.write("\n########################\n")
    print(f"\nO resumo do processo da CrewAI foi salvo em '{output_summary_file}'.")
