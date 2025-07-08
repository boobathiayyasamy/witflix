import os
from mlx_whisper import transcribe

WHISPER_MODEL = "mlx-community/whisper-medium"
DATA_DIR = "data"

class AudioTranscriber:
    @staticmethod
    def list_files(directory=DATA_DIR):
        files = [f for f in os.listdir(directory) if f.lower().endswith(".mp3")]
        return sorted(files)

    def transcribe_audio(self, audio_path):
        print(f"Transcribing audio from {audio_path}")
        result = transcribe(audio_path, path_or_hf_repo=WHISPER_MODEL)
        full_text = ' '.join(segment['text'] for segment in result['segments'])
        for segment in result['segments']:
            print(f"[{segment['start']} – {segment['end']}] {segment['text']}")
        return full_text 