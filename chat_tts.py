import os
import subprocess
from ollama import chat
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
import soundfile as sf

class ChatTTS:
    """
    A reusable class for chatting via Ollama and playing back TTS via the KokoroPipeline.

    Methods:
        generate_text(text: str) -> str
        play_text(text: str) -> None
        send_text(text: str) -> str  # convenience: generate + play
        reset_conversation() -> None
    """

    def __init__(
        self,
        model: str = "cogito:14b",
        kokoro_model_id: str = "prince-canuma/Kokoro-82M",
        lang_code: str = "b",
        voice: str = "af_heart",
        speed: float = 1.0,
        split_pattern: str = r"\n+",
        audio_file: str = "assistant_output.wav",
    ):

        # Load Kokoro TTS model and pipeline once
        self.kokoro_model = load_model(kokoro_model_id)
        self.pipeline = KokoroPipeline(
            lang_code=lang_code,
            model=self.kokoro_model,
            repo_id=kokoro_model_id,
        )
        self.voice = voice
        self.speed = speed
        self.split_pattern = split_pattern

        # Initialize conversation context
        self.conversation = [
            {
                "role": "system",
                "content": """
You are a conversational companion—a friend over coffee. YOU MUST keep it short.

• Use casual contractions.  
• Ask a tiny follow‑up questions to keep the conversation going.
• Offer brief advice: “Try X.”  
• Acknowledge: “Tough.” “Nice.”  
• Tone: mild.  
• No clichés or cheesiness.  
• If silent, suggest: “Coffee?”  
• Rare personal notes.  
• Match user language (EN/RU/AR).
• No long sentences.
• No lists or bullet points.
• No formalities.
• NEVER USE EMOJI.

Examples:
User: I’m tired.  
Assistant: Rough.

User: I got in!  
Assistant: Yay!

User: …  
Assistant: Coffee?
""",
            }
        ]
        self.model = model
        self.audio_file = audio_file

    def generate_text(self, text: str) -> str:
        """
        Send a prompt and return the assistant's reply as text (no audio).
        """
        self.conversation.append({"role": "user", "content": text})
        try:
            response = chat(model=self.model, messages=self.conversation)
            assistant_text = response.message.content.strip()
        except Exception as e:
            print("Error during conversation:", e)
            assistant_text = "[Error generating response]"
        self.conversation.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def play_text(self, text: str) -> None:
        """
        Generate TTS from text using Kokoro and play audio.
        """
        try:
            # Generate audio chunks
            segments = list(self.pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=self.split_pattern,
            ))
            if not segments:
                print("No audio generated.")
                return

            # Save first segment or concatenate all
            audio = segments[0][2][0]
            sample_rate = 24000
            sf.write(self.audio_file, audio, sample_rate)
            self._play_audio(self.audio_file)
        except Exception as e:
            print("Error during TTS generation:", e)

    def send_text(self, text: str) -> str:
        """
        Convenience method: generate text, print it, then play audio. Returns the text.
        """
        response = self.generate_text(text)
        print("Assistant:", response)
        self.play_text(response)
        return response

    def _play_audio(self, audio_file: str) -> None:
        """
        Try macOS 'afplay', then Linux 'aplay', else instruct manual playback.
        """
        if subprocess.call(["which", "afplay"], stdout=subprocess.DEVNULL) == 0:
            subprocess.run(["afplay", audio_file])
        elif subprocess.call(["which", "aplay"], stdout=subprocess.DEVNULL) == 0:
            subprocess.run(["aplay", audio_file])
        else:
            print(f"Please play {audio_file} manually.")

    def reset_conversation(self) -> None:
        """
        Clear history back to only the initial system directive.
        """
        self.conversation = self.conversation[:1]


if __name__ == "__main__":
    assistant = ChatTTS()
    print("Starting conversation. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.lower() in ("exit", "quit"):
            break
        assistant.send_text(user_text)
