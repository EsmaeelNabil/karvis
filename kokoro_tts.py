import os
import subprocess
from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model
import soundfile as sf
from printer import ProgressPrinter

pp = ProgressPrinter()


class KokoroTTS:
    """
    Handles text-to-speech generation and playback using the Kokoro model.
    """

    def __init__(
        self,
        kokoro_model_id: str = "prince-canuma/Kokoro-82M",
        lang_code: str = "b",
        voice: str = "af_heart",
        speed: float = 1.3,
        split_pattern: str = r"\n+",  # Pattern used internally by pipeline if needed
        audio_file: str = "assistant_output.wav",
    ):
        """
        Initializes the Kokoro TTS handler and loads the model.

        Args:
            kokoro_model_id (str): Identifier for the Kokoro model repository.
            lang_code (str): Language code for TTS generation.
            voice (str): The specific voice to use.
            speed (float): Speech playback speed multiplier.
            split_pattern (str): Regex pattern to split text for segment generation.
                                 (Note: often handled within the pipeline itself)
            audio_file (str): Path to save the generated audio file.
        """
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self.split_pattern = split_pattern  # Passed to pipeline during generation
        self.audio_file = audio_file
        self.sample_rate = 24000  # Common sample rate for Kokoro

        # Load Kokoro TTS model and pipeline once
        try:
            pp.progress(f"Loading Kokoro model: {kokoro_model_id}...")
            self.kokoro_model = load_model(kokoro_model_id)
            self.pipeline = KokoroPipeline(
                lang_code=self.lang_code,
                model=self.kokoro_model,
                repo_id=kokoro_model_id,
            )
            pp.progress("Kokoro model loaded successfully.")
        except Exception as e:
            pp.progress(f"Error loading Kokoro model ({kokoro_model_id}): {e}")
            self.pipeline = None  # Mark pipeline as unavailable

    def play_text(self, text: str) -> None:
        """
        Generates TTS audio from the given text using Kokoro and plays it back.

        Args:
            text (str): The text to synthesize into speech.
        """
        if not self.pipeline:
            pp.progress("TTS pipeline not available due to loading error.")
            return
        if not text:
            pp.progress("No text provided for TTS.")
            return

        try:
            pp.progress("Generating TTS audio...")
            for _, _, audio in self.pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=self.split_pattern,
            ):
                sf.write(self.audio_file, audio[0], self.sample_rate)
                pp.progress(f"Audio saved to {self.audio_file}")
                self._play_audio(self.audio_file)

        except Exception as e:
            pp.progress(f"Error during TTS generation or playback: {e}")

    def _play_audio(self, audio_file: str) -> None:
        """
        Internal method to play the generated audio file using system commands.
        Tries macOS 'afplay', then Linux 'aplay'. Warns if neither is found.
        """
        if not os.path.exists(audio_file):
            pp.progress(f"Audio file not found: {audio_file}")
            return

        try:
            # Check for afplay (macOS)
            if (
                subprocess.call(
                    ["which", "afplay"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                == 0
            ):
                pp.progress("Playing audio using afplay...")
                subprocess.run(["afplay", audio_file], check=True)
            # Check for aplay (Linux/ALSA)
            elif (
                subprocess.call(
                    ["which", "aplay"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                == 0
            ):
                pp.progress("Playing audio using aplay...")
                subprocess.run(["aplay", audio_file], check=True)
            else:
                pp.progress(
                    "Neither 'afplay' nor 'aplay' found. Please play the audio file manually:"
                )
                pp.progress(f"Audio file saved at: {os.path.abspath(audio_file)}")
        except subprocess.CalledProcessError as e:
            pp.progress(f"Error playing audio: {e}")
        except FileNotFoundError:
            pp.progress(
                "Error: 'which' command not found. Cannot determine audio player."
            )
        except Exception as e:
            pp.progress(f"An unexpected error occurred during audio playback: {e}")
