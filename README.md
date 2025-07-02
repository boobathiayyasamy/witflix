# WitFlix 

Streaming sass, one quote at a time - A lightweight Gradio app to critique audio clips using MLX and Whisper.

## Description

WitFlix is a demo application that showcases Apple's MLX framework combined with Whisper and Llama 3.2 models. 
The app allows you to upload audio clips, transcribe them using Whisper, and then generate lighthearted 
comments about the content using Llama 3.2. The app has been crafted to work best with short audio clips from films. 

You can download audio clips from [moviesoundclips.net](https://www.moviesoundclips.net/) or, for a bit more fun, you 
could also record an audio clip.

Key features:
- Audio transcription using MLX-optimised Whisper model
- Text summarisation using MLX-optimised Llama 3.2 model
- Simple and intuitive Gradio web interface
- Optimised for Apple Silicon (M1/M2/M3) Macs

## Getting Started

### Prerequisites

- Python 3.12 or higher
- A Hugging Face account and API token
- macOS with Apple Silicon (M1/M2/M3) for optimal performance
- The [Homebrew](https://brew.sh/) package manager 

### Preparation

#### Install `uv`

This project uses `uv` for dependency management. If you don't have `uv` installed, you can install it following the 
instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

#### Install `ffmpeg`

To use speech recognition with [Whisper in MLX](https://github.com/ml-explore/mlx-examples/tree/main/whisper), 
you need to install `ffmpeg`

```bash
brew install ffmpeg
```

### Clone the Repository

Duh!

```bash
git clone https://github.com/bennettsmj/witflix.git
cd witflix
```

### Environment Setup with UV

1. Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e . # optional  
```

2. Create a `.env` file in the project root with your Hugging Face token:

```bash
echo "HF_TOKEN=your_huggingface_token" > .env
```

Replace `your_huggingface_token` with your actual Hugging Face token. You can obtain a token from your [Hugging Face account settings](https://huggingface.co/settings/tokens).

### Running the Demo

To run the demo:

```bash
uv run main.py
```

This will start the Gradio web interface, which you can access in your browser (typically at http://127.0.0.1:7860).

## Usage

1. Upload or record an audio clip using the Gradio interface
2. Click "Transcribe audio" to generate a text transcription
3. Click "Summarise" to get a lighthearted comment about the clip

The app comes with a sample audio clip from "Life of Brian" in the `data` directory that you can use for testing.

## Dependencies

- gradio: Web interface
- mlx, mlx-lm, mlx-whisper: Apple's ML framework and models
- huggingface-hub: For accessing models
- python-dotenv: For loading environment variables
- torch, torchaudio, torchvision: PyTorch dependencies

## License

This project is licensed under the terms included in the LICENSE file.
