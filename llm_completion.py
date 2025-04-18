from ollama import chat

class OllamaCompletion:
    """
    Handles text generation using an Ollama model, maintaining conversation history.
    """
    DEFAULT_SYSTEM_PROMPT = """
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
"""

    def __init__(
        self,
        model: str = "cogito:14b",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        """
        Initializes the Ollama completion handler.

        Args:
            model (str): The Ollama model identifier to use.
            system_prompt (str): The initial system prompt to guide the AI's behavior.
        """
        self.model = model
        self.system_prompt_content = system_prompt.strip()
        self.conversation = [
            {
                "role": "system",
                "content": self.system_prompt_content,
            }
        ]

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