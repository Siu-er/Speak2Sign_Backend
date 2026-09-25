"""Azure OpenAI text stages: reconstructing English from a recognized gloss
sequence, and translating a spoken utterance into English."""

import os

_client = None

_SYSTEM_PROMPT = (
    "You reconstruct one natural, first-person English sentence from an ordered "
    "list of recognized American Sign Language glosses. Each gloss is a sign the "
    "person actually made - incorporate the meaning of EVERY gloss, do not drop "
    "any. ASL drops pronouns, articles, the verb 'to be' and tense, and reorders "
    "words; restore them. Notes: the gloss 'sign' means sign language or signing; "
    "a stray 'yes'/'no' may be a filler - fold it in only if natural. Recognition "
    "is imperfect, so read the glosses as a whole and produce the most plausible "
    "everyday sentence. Reply with ONLY the sentence, no quotes or notes."
)

_TRANSLATE_PROMPT = (
    "You translate a spoken utterance into English. The utterance may be in any "
    "language and is conversational speech, so keep the register informal and the "
    "meaning exact. Reply with ONLY the English translation, no quotes or notes. "
    "If the utterance is already English, repeat it unchanged."
)


def _get_client():
    global _client
    if _client is None:
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
            api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
        )
    return _client


def _complete(system_prompt, user_content, max_tokens):
    """Raises if the LLM is not configured or returns nothing (no silent fallback)."""
    deployment = os.environ.get("AZURE_OPENAI_GENERATOR_DEPLOYMENT")
    if not deployment or not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("LLM not configured")
    resp = _get_client().chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("LLM returned empty output")
    return text


def gloss_to_sentence(signs):
    gloss = " ".join(str(s).strip() for s in signs if str(s).strip())
    if not gloss:
        raise ValueError("empty gloss sequence")
    return _complete(_SYSTEM_PROMPT, gloss, 60)


def translate_to_english(utterance):
    text = utterance.strip()
    if not text:
        raise ValueError("empty utterance")
    return _complete(_TRANSLATE_PROMPT, text, 80)
