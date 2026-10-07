import json
import sys
from typing import Any
from groq import Groq
from pydantic import BaseModel
from rich.console import Console
from cpc_agent.config import GROQ_API_KEY, GROQ_MODEL

console = Console(stderr=True)
_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            console.print(
                "[bold red]ERROR:[/bold red] GROQ_API_KEY is not set.\n"
                "Add it to your .env file or export it: [bold]export GROQ_API_KEY=...[/bold]\n"
                "Get a free key at https://console.groq.com/keys"
            )
            sys.exit(1)
        # Free tier has tight per-minute token limits: let the SDK back off on 429s.
        _client = Groq(api_key=GROQ_API_KEY, max_retries=6)
    return _client


def health_check(model: str | None = None) -> None:
    target = model or GROQ_MODEL
    client = get_client()
    try:
        client.chat.completions.create(
            model=target,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as e:
        console.print(
            f"[bold red]ERROR:[/bold red] Cannot reach Groq with model '{target}'.\n"
            f"Check GROQ_API_KEY and that the model is available on your account.\n"
            f"Details: {e}"
        )
        sys.exit(1)


def call_json(
    prompt: str,
    schema: type[BaseModel],
    model: str | None = None,
    system: str | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Call Groq with JSON mode, parse and validate against schema. Returns raw dict."""
    target = model or GROQ_MODEL
    client = get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    messages[-1]["content"] += f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_json}"

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=target,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            parsed = schema.model_validate_json(raw)
            return parsed.model_dump()
        except Exception as e:
            if attempt < max_retries:
                console.print(f"[yellow]LLM retry {attempt + 1}/{max_retries}: {e}[/yellow]")
            else:
                raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {e}") from e

    raise RuntimeError("Unreachable")


def call_text(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
) -> str:
    """Simple text call to Groq, returns string content."""
    target = model or GROQ_MODEL
    client = get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=target,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()
