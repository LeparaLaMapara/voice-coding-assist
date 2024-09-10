
# Voice Coding Assistant - Version 2

This updated version of the Voice Coding Assistant integrates with Jupyter Notebook, allowing you to interact with your assistant in real-time. The assistant listens to your spoken commands, generates Python code, executes it within the notebook, and handles errors dynamically.

## Components

1. **Whisper Model**: For real-time speech-to-text transcription using your microphone.
2. **OpenAI GPT**: For converting commands into executable Python code and providing error suggestions.
3. **Jupyter Notebook Integration**: Allows the generated code to be executed directly within the notebook.
4. **PyAudio**: For capturing audio input in real-time.

## Installation

### Dependencies

Ensure you have Python 3.8+ installed. Use the following command to install all required dependencies:

```bash
pip install torch torchvision torchaudio openai whisper flask pyaudio requests
```

### Setting Up Whisper

```bash
# Install PyTorch (ensure your system has Python 3.8+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Whisper
pip install git+https://github.com/openai/whisper.git
```

### Setting Up OpenAI API

Sign up at [OpenAI](https://platform.openai.com/) and retrieve your API key. Replace `'your-openai-api-key'` in the scripts with your actual key.

## Running the Assistant in Jupyter Notebook

### Step 1: Record and Transcribe Audio

```python
# Import necessary libraries
import whisper
import openai
import pyaudio
import wave
import sys
import io

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

### Step 3: Execute Python Code in Jupyter Notebook

```python
# Function to execute Python code directly in Jupyter Notebook
def execute_python_code_in_notebook(code):
    try:
        exec(code, globals())  # Executes code directly in the current Jupyter environment
    except Exception as e:
        print(f"Error during execution: {e}")
```

### Full Interactive Workflow in Jupyter Notebook

```python
# Full Demo Run
transcribed_command = record_and_transcribe()
print(f"Transcribed Command: {transcribed_command}")

generated_code = get_code_from_command(transcribed_command)
print(f"Generated Code:
{generated_code}")

# Execute the generated code in the notebook environment
execute_python_code_in_notebook(generated_code)
```

## Usage

Run each of the steps in sequence in a Jupyter Notebook to see the assistant in action. Speak your commands, watch the code generate, and see the execution results live in the notebook.

## License

Open source and available for modification to fit your needs. Contributions are welcome!
