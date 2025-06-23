%%writefile tools.py
# tools.py

import os
from crewai_tools import BaseTool 
from moviepy.editor import VideoFileClip
from transformers import pipeline
import torch

class ColabVideoToAudioTool(BaseTool):
    name: str = "Colab Video to Audio Converter"
    description: str = "Converts a video file (e.g., MP4) into an MP3 audio file within the Google Colab environment. " \
                       "Requires full input and output file paths. " \
                       "Ensure moviepy is installed (`!pip install moviepy`)."

    def _run(self, video_path: str, audio_output_path: str) -> str:
        try:
            if not os.path.exists(video_path):
                return f"Erro: Arquivo de vídeo de entrada não encontrado em '{video_path}'."
            output_dir = os.path.dirname(audio_output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_output_path, codec='mp3')
            audio_clip.close()
            video_clip.close()
            return f"Sucesso: Vídeo '{video_path}' convertido para áudio MP3 em '{audio_output_path}'."
        except Exception as e:
            return f"Erro durante a conversão de vídeo para áudio: {e}"


class ColabWhisperTranscriptionTool(BaseTool):
    name: str = "Colab Whisper Audio Transcriber"
    description: str = "Transcribes an audio file (e.g., MP3) into text using the Whisper ASR model " \
                       "within the Google Colab environment. Requires full input audio file path and " \
                       "output text file path. " \
                       "Ensure transformers, accelerate, optimum, bitsandbytes are installed."

    def _run(self, audio_path: str, transcription_output_path: str, model_name: str = "openai/whisper-small") -> str:
        try:
            if not os.path.exists(audio_path):
                return f"Erro: Arquivo de áudio de entrada não encontrado em '{audio_path}'."
            output_dir = os.path.dirname(transcription_output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Usando dispositivo: {device} para Whisper.")
            transcriber = pipeline("automatic-speech-recognition", model=model_name, device=device)
            transcription = transcriber(audio_path)["text"]
            with open(transcription_output_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            return f"Sucesso: Áudio '{audio_path}' transcrito para '{transcription_output_path}'. Transcrição: '{transcription[:200]}...' (primeiros 200 caracteres)."
        except Exception as e:
            return f"Erro durante a transcrição do áudio com Whisper: {e}"


class FileManagerTool(BaseTool):
    name: str = "File Manager"
    description: str = "Manages file operations such as checking existence, reading content, and writing content to files."

    def _run(self, operation: str, path: str, content: str = None) -> str:
        try:
            if operation == 'exists':
                return str(os.path.exists(path))
            elif operation == 'read':
                if not os.path.exists(path):
                    return f"Erro: Arquivo não encontrado para leitura em '{path}'."
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            elif operation == 'write':
                output_dir = os.path.dirname(path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Sucesso: Conteúdo escrito em '{path}'."
            elif operation == 'makedirs':
                os.makedirs(path, exist_ok=True)
                return f"Sucesso: Diretório '{path}' criado ou já existente."
            else:
                return f"Erro: Operação de arquivo '{operation}' não suportada."
        except Exception as e:
            return f"Erro na operação de arquivo '{operation}' em '{path}': {e}"
