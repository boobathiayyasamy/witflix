import os
import gradio as gr
from mlx_lm import load, generate
from mlx_whisper import transcribe
from dotenv import load_dotenv
from huggingface_hub import login

DATA_DIR = "data"
AUDIO_FILE = f"{DATA_DIR}/life_of_brian.mp3"
WHISPER_MODEL = "mlx-community/whisper-medium"
LLM_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

def transcribe_audio(audio_path):
    """
    Transcribe an audio file using a pre-defined Whisper model.

    This function leverages the Whisper model to convert the input audio file
    specified by the `audio_path` into its transcribed text form. The audio
    file path provided must point to a valid audio format supported by the
    Whisper library.

    :param audio_path: The file path to the audio file to be transcribed.
    :type audio_path: Str
    :return: The transcribed text from the audio file.
    :rtype: Str
    """
    print(f"Transcribing audio from {audio_path}")
    result = transcribe(audio_path, path_or_hf_repo=WHISPER_MODEL)
    # Concatenate all segment texts
    full_text = ' '.join(segment['text'] for segment in result['segments'])
    for segment in result['segments']:
        print(f"[{segment['start']} – {segment['end']}] {segment['text']}")
    return full_text


# Summarise transcription
def summarize_transcription(transcription):
    """
    Summarises the provided transcription by using a language model to extract key
    information and present it concisely.

    :param transcription: The original transcription of an audio file that needs to
        be summarised.
    :type transcription: Str
    :return: A summarised version of the given transcription.
    :rtype: Any
    """
    system_prompt = "You are a film buff, not a critic. "
    user_prompt = (f"The following is an audio transcript of a clip from the move. Write a very short, lighthearted "
                   f"comment on the content of the clip. For example, "
                   f"\"it's a hilarious satirical comedy that'll have you in stitches\" or, "
                   f"\"it's an action packed adventure that'll keep you on the edge of your seat.\"\n"
                   f"Don't try to reference the film in anyway."
                   f"\n\n{transcription}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print("Loading model and tokenizer, and generating summary from transcription...")
    model, tokenizer = load(LLM_MODEL)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return generate(model, tokenizer, prompt, verbose=True)

def list_files():
    files = [f for f in os.listdir("data") if f.lower().endswith(".mp3")]
    return sorted(files)


def main():
    # Load environment variable from .env
    load_dotenv()

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

        b1.click(transcribe_audio, inputs=audio_file, outputs=text)
        b2.click(summarize_transcription, inputs=text, outputs=label)

    demo.launch()

if __name__ == "__main__":
    main()
