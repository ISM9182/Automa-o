%%writefile tasks.py
# tasks.py

from crewai import Task
from textwrap import dedent

class CustomTasks:
    def __init__(self):
        pass

    def process_media_task(self, agent, video_input_path, audio_output_path):
        return Task(
            description=dedent(f"""
                Converta o arquivo de vídeo localizado em '{video_input_path}' para um arquivo de áudio MP3 de alta qualidade.
                O arquivo de áudio resultante deve ser salvo como '{audio_output_path}'.
                Garanta que todas as operações de arquivo sejam realizadas de forma robusta e eficiente no ambiente do Google Colab.
                Verifique se o arquivo de vídeo existe antes de tentar a conversão.
            """),
            expected_output=dedent(f"""
                O caminho completo para o arquivo MP3 gerado: '{audio_output_path}'.
                Uma confirmação clara de que o processo de conversão foi concluído com sucesso e o arquivo MP3 está pronto.
                Se houver um erro, uma descrição detalhada do motivo da falha.
            """),
            agent=agent,
        )

    def transcribe_audio_task(self, agent, audio_input_path, transcription_output_path):
        return Task(
            description=dedent(f"""
                Transcreva o conteúdo do arquivo de áudio MP3 localizado em '{audio_input_path}' para texto.
                Salve a transcrição completa e precisa no arquivo '{transcription_output_path}'.
                Utilize as melhores práticas para otimização do reconhecimento de fala, garantindo alta fidelidade e clareza.
                Certifique-se de que o áudio de entrada existe antes de iniciar a transcrição.
            """),
            expected_output=dedent(f"""
                O caminho completo para o arquivo de texto com a transcrição: '{transcription_output_path}'.
                A transcrição completa e revisada do áudio, pronta para ser usada para criação de conteúdo didático.
                Se houver um erro, uma descrição detalhada do motivo da falha.
            """),
            agent=agent,
        )

    def didactic_content_creation_task(self, agent, transcription_input_path, html_output_path):
        return Task(
            description=dedent(f"""
                Baseado na transcrição textual do arquivo '{transcription_input_path}', crie um resumo didático e explicativo.
                Organize o conteúdo por tópicos de forma clara, coesa e envolvente para o público-alvo.
                Formate o resumo final em HTML, salvando-o em '{html_output_path}', para torná-lo um material educacional
                acessível e interativo. O HTML deve ser bem estruturado com cabeçalhos (h1, h2, etc.), parágrafos,
                e listas (ordenadas/não ordenadas) quando apropriado.
                Garanta que a transcrição de entrada possa ser lida e que o arquivo HTML seja salvo corretamente.
            """),
            expected_output=dedent(f"""
                Um arquivo HTML didático e bem formatado com o resumo completo do conteúdo, localizado em '{html_output_path}'.
                O HTML deve ser de alta qualidade, pronto para ser usado como material educacional.
                Se houver um erro, uma descrição detalhada do motivo da falha.
            """),
            agent=agent,
        )
