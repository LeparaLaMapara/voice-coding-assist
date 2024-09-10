
# Voice Coding Assistant Guide

## Overview

This guide walks you through setting up a voice-activated coding assistant that listens to spoken commands, generates Python code, executes it in real-time, and handles errors. The assistant uses Whisper for speech-to-text transcription and OpenAI's GPT model for generating and debugging code.

## Components

1. **Whisper Model**: For real-time speech-to-text transcription.
2. **OpenAI GPT**: For converting commands to code and suggesting fixes for errors.
3. **Flask Backend**: To manage the execution environment for the code.
4. **PyAudio**: For capturing audio input in real-time.

## Installation

### Dependencies

Ensure you have Python 3.8+ installed. Install the required dependencies using:

```bash
pip install torch torchvision torchaudio openai whisper flask pyaudio requests
```

### Setting Up Whisper

```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Whisper
pip install git+https://github.com/openai/whisper.git
```

### Setting Up OpenAI API

Sign up at [OpenAI](https://platform.openai.com/) and retrieve your API key. Replace `'your-openai-api-key'` in the scripts with your actual key.

## Step-by-Step Guide

### Step 1: Record and Transcribe Audio

```python
import whisper
import pyaudio
import wave

# Load Whisper model
model = whisper.load_model("base")

# Function to record and transcribe audio in real-time
def record_and_transcribe():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "live_demo.wav"

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Transcribe audio using Whisper
    result = model.transcribe(WAVE_OUTPUT_FILENAME)
    return result['text']
```

### Step 2: Convert Command to Code Using GPT

```python
import openai

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'  # Replace with your actual API key

# Function to convert text command into Python code using OpenAI
def get_code_from_command(command):
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Convert the following command into Python code:

Command: {command}

Python Code:",
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].text.strip()
```

### Step 3: Execute and Debug Code

```python
import sys
import io

# Function to execute code and use OpenAI for error suggestion if needed
def execute_and_debug(code):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(code)
        output = sys.stdout.getvalue()
    except Exception as e:
        error_message = str(e)
        suggestion = openai.Completion.create(
            model="gpt-4",
            prompt=f"The following code caused an error:

{code}

Error: {error_message}

Suggest a fix:",
            max_tokens=150,
            temperature=0.5
        ).choices[0].text.strip()
        output = f"Error: {error_message}

Suggested Fix:
{suggestion}"
    finally:
        sys.stdout = old_stdout
    return output
```

### Full Demo Workflow

```python
# Full Demo Run
transcribed_text = record_and_transcribe()
print(f"Transcribed Command: {transcribed_text}")

generated_code = get_code_from_command(transcribed_text)
print(f"Generated Code:
{generated_code}")

execution_result = execute_and_debug(generated_code)
print(f"Execution Result:
{execution_result}")
```

## Usage

Run each of the steps in sequence in a Jupyter Notebook or Python environment to see the assistant in action.

## License

Open source and available for modification to fit your needs. Contributions are welcome!
