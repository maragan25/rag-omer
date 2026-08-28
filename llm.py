# llm.py — Sends augmented prompts to the local LLM via Foundry Local

from foundry_local_sdk.openai import ChatClientSettings
from foundry_client import get_ready_model
from config import LLM_MODEL, SYSTEM_PROMPT

_client = get_ready_model(LLM_MODEL).get_chat_client()
_client.settings = ChatClientSettings(temperature=0.2, max_tokens=512)

def generate_answer(question: str, context: str, history: list[tuple[str, str]] | None = None) -> str:
    """
    Build an augmented prompt from the question + retrieved context,
    then call the local LLM and return its response text.

    history: recent (question, answer) turns, oldest first, replayed as prior
    chat turns so the model can resolve follow-ups ("when?" after "who
    invented C?") without re-sending their retrieved context.
    """
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for prev_question, prev_answer in (history or []):
        messages.append({"role": "user", "content": prev_question})
        messages.append({"role": "assistant", "content": prev_answer})
    messages.append({"role": "user", "content": user_message})

    response = _client.complete_chat(messages=messages)

    return response.choices[0].message.content.strip()