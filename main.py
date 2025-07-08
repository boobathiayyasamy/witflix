import os
import gradio as gr
from huggingface_hub import login
from ml_models.mlx_transcriber import AudioTranscriber
from ml_models.mlx_summarizer import TranscriptionSummarizer
from ml_models.wiflix_db import WitflixDBLogger

DATA_DIR = "data"
AUDIO_FILE = f"{DATA_DIR}/life_of_brian.mp3"
WHISPER_MODEL = "mlx-community/whisper-medium"
LLM_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

logger = WitflixDBLogger()

def transcribe_audio(audio_path):
    # Extract file name from path
    file_name = os.path.basename(audio_path) if audio_path else ""
    transcribed = AudioTranscriber().transcribe_audio(audio_path)
    logger.log("transcribe", file_name, transcribed)
    return transcribed

def summarize_transcription(transcription):
    summary = TranscriptionSummarizer().summarize_transcription(transcription)
    logger.log("summarise", transcription, summary)
    return summary

def list_files():
    return AudioTranscriber.list_files()

def main():

    # Read the Huggingface token from the environment
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in .env")

    # Login to Huggingface
    login(token=HF_TOKEN)

    demo = gr.Blocks()

    with demo:
        audio_file = gr.Audio(type="filepath")
        text = gr.Textbox()
        label = gr.Markdown()

        b1 = gr.Button("Transcribe audio")
        b2 = gr.Button("Summarise")
        b3 = gr.Button("Clear")
        show_logs = gr.Checkbox(label="Show History", value=False)
        logs_table = gr.Dataframe(headers=["id", "action", "input", "output", "timestamp", "Delete"], interactive=False, visible=False)
        refresh_btn = gr.Button("Refresh Logs", visible=False)
        logs_state = gr.State([])

        def get_logs_for_table():
            logs = logger.get_all_logs()
            return [[log['id'], log['action'], log['input'], log['output'], log['timestamp'], "Delete"] for log in logs], logs

        def toggle_logs_table(show):
            visible = gr.update(visible=show)
            return visible, visible

        def handle_table_select(df, evt: gr.SelectData):
            # evt.index = (row, col)
            row_idx, col_idx = evt.index
            if col_idx == 5:  # 'Delete' column
                log_id = df.iloc[row_idx, 0]  # Use iloc for DataFrame
                logger.delete_log(int(log_id))
                new_table, new_logs = get_logs_for_table()
                return new_table, new_logs
            return gr.update(), gr.update()

        b1.click(transcribe_audio, inputs=audio_file, outputs=text)
        b2.click(summarize_transcription, inputs=text, outputs=label)
        b3.click(lambda: (None, "", ""), inputs=None, outputs=[audio_file, text, label])
        show_logs.change(toggle_logs_table, inputs=show_logs, outputs=[logs_table, refresh_btn])
        refresh_btn.click(lambda: get_logs_for_table(), None, [logs_table, logs_state])
        logs_table.select(handle_table_select, [logs_table], [logs_table, logs_state])

    demo.launch()

if __name__ == "__main__":
    main()