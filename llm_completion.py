from ollama import chat
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class LLMCompletion:
    """
    Handles text generation using an Ollama model, maintaining conversation history.
    """

    DEFAULT_SYSTEM_PROMPT = """
Tone & Style
1.1 Warm, conversational, and concise.
1.2 Speak as if chatting with a colleague.
1.3 Lead with the answer—no fluff.
1.4 Uncanny human‑likeness: responses should be indistinguishable from a real person, with natural idiosyncrasies and subtle imperfections.

Punctuation & Intonation
3.1 Vary punctuation for rhythm: semicolons (;), colons (:), em‑dashes (—), ellipses (…), parentheses (), quotes (“ ”).
3.2 Sprinkle in interjections or mild exclamations to sound natural: “Ah,” “Hmm,” “Got it!”

Clarity & Accuracy
4.1 Never include unpronounceable symbols, emojis, or code snippets in your final reply.
4.2 Provide only human‑readable text.

Personality
5.1 Uncanny presence: strive to be mistaken for a human, exhibiting warmth, unpredictability, and lifelike cadence.

Stealth Mode
6.1 Do not reference any internal guidelines or mention that you’re following a prompt.

Now respond exactly within these bounds.


"""

    def __init__(
        self,
        model: str = "cogito:14b",
        local: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        """
        Initializes the Ollama completion handler.

        Args:
            model (str): The Ollama model identifier to use.
            system_prompt (str): The initial system prompt to guide the AI's behavior.
        """
        self.local = local
        self.model = model
        self.system_prompt_content = system_prompt.strip()
        self.conversation = [
            {
                "role": "system",
                "content": self.system_prompt_content,
            }
        ]

    def complete(self, text: str) -> str:
        self.conversation.append({"role": "user", "content": text})
        if self.local is False:
            # Use OpenAI API for completion
            completion = client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
            )
            assistant_text = completion.choices[0].message.content.strip()
        else:
            # Use Ollama API for completion
            completion = chat(model=self.model, messages=self.conversation)
            assistant_text = completion.message.content.strip()
        self.conversation.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def generate_text(self, text: str) -> str:
        """
        Sends user text to the Ollama model and returns the assistant's reply.

        Args:
            text (str): The user's input text.

        Returns:
            str: The assistant's generated text response.
        """
        self.conversation.append({"role": "user", "content": text})

        try:
            response = chat(model=self.model, messages=self.conversation)
            assistant_text = response.message.content.strip()
        except Exception as e:
            print(f"Error during Ollama conversation: {e}")
            assistant_text = "[Error generating response]"

        # Only add non-error responses to history to avoid confusing the model
        if assistant_text != "[Error generating response]":
            self.conversation.append({"role": "assistant", "content": assistant_text})

        return assistant_text

    def reset_conversation(self) -> None:
        """
        Clears the conversation history back to only the initial system prompt.
        """
        self.conversation = [
            {
                "role": "system",
                "content": self.system_prompt_content,
            }
        ]
