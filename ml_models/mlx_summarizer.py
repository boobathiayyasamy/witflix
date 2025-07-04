from mlx_lm import load, generate

LLM_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

class TranscriptionSummarizer:
    def summarize_transcription(self, transcription):
        system_prompt = "You are a film buff, not a critic. "
        user_prompt = (f"The following is an audio transcript of a clip from the move. Write a very short, lighthearted "
                       f"comment on the content of the clip. For example, "
                       f'"it\'s a hilarious satirical comedy that\'ll have you in stitches" or, '
                       f'"it\'s an action packed adventure that\'ll keep you on the edge of your seat."\n'
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