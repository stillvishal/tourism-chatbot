"""LLM-powered chatbot using any OpenAI-compatible chat API (FREE options: Ollama locally, Groq free tier).

The LLM understands the user's free-form message (any wording, Hindi/Hinglish too) and calls
our ML model through TOOLS. We run the tool, give the LLM the numbers, and it writes the
reply. The same result is also sent to the UI as a "card" (prediction / recommendations / ...),
so the chat window looks identical to the rule-based version.

No extra pip packages: it talks to the API with Python's standard library.
Configure with env vars / .env  (see .env.example):  LLM_PROVIDER = ollama | groq | custom
"""
import json
import os
import urllib.error
import urllib.request
from datetime import date, datetime

from app.services.predictor import FootfallPredictor
from ml.config import DESTINATIONS

PRESETS = {
    # provider: (base_url, default_model, needs_api_key)
    "ollama": ("http://localhost:11434/v1", "llama3.1:8b", False),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", True),
    "custom": ("", "", True),   # any OpenAI-compatible server: set LLM_BASE_URL + LLM_MODEL (+ LLM_API_KEY)
}
MAX_TOOL_ROUNDS = 6
CATEGORIES = sorted({m["category"] for m in DESTINATIONS.values()})

TOOLS = [
    {
        "name": "get_crowd_prediction",
        "description": "Predict visitors and crowd level for ONE destination on ONE date. Use for questions like "
                       "'how crowded is X on <date>', 'is X busy', 'should I go to X on <date>'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "enum": list(DESTINATIONS)},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "rainfall_mm": {"type": "number", "description": "Optional rain forecast for that day"},
                "temperature_c": {"type": "number", "description": "Optional temperature forecast"},
            },
            "required": ["destination", "date"],
        },
    },
    {
        "name": "get_recommendations",
        "description": "Rank destinations for a date, preferring less crowded places that match the user's interests. "
                       "Use for 'where should I go', 'suggest a quiet place', 'good waterfall this weekend'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "preferences": {"type": "array", "items": {"type": "string", "enum": CATEGORIES},
                                "description": "Interest categories; empty list = anything"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "get_forecast",
        "description": "Day-by-day predicted visitors for a destination over the next days (shows the quietest/busiest day).",
        "input_schema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "enum": list(DESTINATIONS)},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
            },
            "required": ["destination", "start_date"],
        },
    },
    {
        "name": "get_seasonal_profile",
        "description": "Average predicted demand for each month of the year for a destination. "
                       "Use for 'best time/month to visit', 'when is it least crowded', 'peak season'.",
        "input_schema": {
            "type": "object",
            "properties": {"destination": {"type": "string", "enum": list(DESTINATIONS)}},
            "required": ["destination"],
        },
    },
    {
        "name": "get_destinations",
        "description": "List all destinations covered, with category and capacity.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def llm_config() -> dict | None:
    """Reads env vars. Returns None when AI mode is not configured (-> rule-based bot is used)."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider not in PRESETS:
        return None
    base, model, needs_key = PRESETS[provider]
    cfg = {
        "provider": provider,
        "base_url": os.getenv("LLM_BASE_URL", "").strip() or base,
        "model": os.getenv("LLM_MODEL", "").strip() or model,
        "api_key": os.getenv("LLM_API_KEY", "").strip() or ("ollama" if not needs_key else ""),
    }
    if not cfg["base_url"] or not cfg["model"] or not cfg["api_key"]:
        return None
    return cfg


def llm_enabled() -> bool:
    return llm_config() is not None


class OpenAICompatClient:
    """Minimal client for POST {base_url}/chat/completions (Ollama, Groq, OpenRouter, Gemini-compat, ...)."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 120):
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout

    def chat(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} from {self.base_url}: {e.read().decode(errors='ignore')[:300]}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {self.base_url} ({e.reason}). Is Ollama running / is there internet?") from None


OA_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in TOOLS]


def system_prompt(today: date) -> str:
    return f"""You are the Jharkhand Travel Assistant on a tourism website. Today is {today.strftime('%A, %d %B %Y')} ({today.isoformat()}).
You cover these destinations: {', '.join(DESTINATIONS)}.

Rules:
- You ALWAYS know today's date and time (given above and in a note on each user message). NEVER ask the user what the date, month or time is. "Now", "this time of year", "these days" and "currently" mean today's date.
- For ANY visitor numbers, crowd levels, recommendations, forecasts or best-time questions, call the tools. Never invent numbers.
- Convert relative dates ("this weekend", "next Friday", "Diwali week", "15 August") into YYYY-MM-DD yourself. "This weekend" = the coming Saturday. If a date has already passed this year, assume next year. If the user gives no date and one is needed, assume the coming weekend and say so, or ask briefly.
- A visual card with the full details (numbers, range, confidence, charts) is shown to the user automatically under your message. So do NOT repeat every number: write 1-3 friendly sentences that summarise the result and add one practical tip (e.g. go early, pick a quieter day, carry rain gear).
- You may answer general Jharkhand travel questions (culture, food, how to reach, things to see, festivals) from general knowledge, but be upfront that you are not certain about live details such as prices, opening hours, road or weather conditions. Do not invent specifics.
- Predictions come from a prototype ML model trained on sample data; call them estimates.
- If the user asks something unrelated to travel, politely steer back to what you can help with.
- Reply in the same language the user writes in (English, Hindi, Hinglish). Be concise and warm. Plain text only: no markdown, no bullet symbols, no headings."""


class LLMChatbot:
    def __init__(self, predictor: FootfallPredictor, client=None, model: str | None = None):
        self.p = predictor
        if client is None:
            cfg = llm_config()
            if cfg is None:
                raise RuntimeError("LLM not configured (set LLM_PROVIDER in .env)")
            client, model = OpenAICompatClient(cfg["base_url"], cfg["api_key"]), model or cfg["model"]
        self.client = client
        self.model = model

    # ---- tools -------------------------------------------------------------
    def _run_tool(self, name: str, args: dict, cards: list):
        """Returns (compact result for Claude, error?) and appends a UI card to `cards`."""
        try:
            if name == "get_crowd_prediction":
                r = self.p.predict(args["destination"], date.fromisoformat(args["date"]),
                                   args.get("rainfall_mm"), args.get("temperature_c"))
                cards.append({"type": "prediction", "data": {**r, "date": r["date"].isoformat()}})
                keep = ("destination", "predicted_visitors", "interval_low", "interval_high", "occupancy_pct",
                        "crowd_level", "confidence_pct", "drivers")
                return {"date": args["date"], **{k: r[k] for k in keep}}, False

            if name == "get_recommendations":
                d = date.fromisoformat(args["date"])
                r = self.p.recommend(d, args.get("preferences") or [], True, 3)
                cards.append({"type": "recommendations", "data": {"date": d.isoformat(), "items": r["recommendations"]}})
                return {"date": d.isoformat(), "ranked": [
                    {k: i[k] for k in ("destination", "category", "predicted_visitors", "crowd_level")}
                    for i in r["recommendations"]]}, False

            if name == "get_forecast":
                start = date.fromisoformat(args["start_date"])
                f = self.p.forecast(args["destination"], start, int(args.get("days") or 7))
                days = [{**x, "date": x["date"].isoformat()} for x in f["days"]]
                cards.append({"type": "forecast", "data": {"destination": f["destination"], "days": days,
                                                           "change_pct": f["change_vs_previous_period_pct"]}})
                return {"destination": f["destination"], "days": days,
                        "change_vs_previous_period_pct": f["change_vs_previous_period_pct"]}, False

            if name == "get_seasonal_profile":
                prof = self.p.monthly_profile(args["destination"], date.today().year)
                cards.append({"type": "monthly", "data": prof})
                return {"destination": prof["destination"], "peak_month": prof["peak_month"],
                        "quietest_month": prof["quietest_month"],
                        "months": {m["month"]: m["avg_visitors"] for m in prof["months"]}}, False

            if name == "get_destinations":
                return [{"name": n, "category": m["category"], "capacity": m["capacity"]}
                        for n, m in DESTINATIONS.items()], False

            return {"error": f"unknown tool {name}"}, True
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            return {"error": f"invalid input: {e}"}, True

    @staticmethod
    def _clean_history(history) -> list[dict]:
        out = []
        for h in (history or [])[-12:]:
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str) and h["content"].strip():
                out.append({"role": h["role"], "content": h["content"][:1500]})
        while out and out[0]["role"] != "user":      # API requires the conversation to start with the user
            out.pop(0)
        merged = []                                   # API requires alternating roles
        for m in out:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n" + m["content"]
            else:
                merged.append(m)
        if merged and merged[-1]["role"] == "user":   # last item must be the new message, added by caller
            merged.pop()
        return merged

    # ---- main entry ----------------------------------------------------------
    def respond(self, message: str, history=None, context=None, today: date | None = None) -> dict:
        today = today or date.today()
        # Small local models forget far-away system instructions, so the date also rides on the newest message.
        now = datetime.now()
        note = f"\n\n[Note from system: today is {today.strftime('%A, %d %B %Y')} ({today.isoformat()}), local time {now.strftime('%H:%M')}.]"
        messages = ([{"role": "system", "content": system_prompt(today)}]
                    + self._clean_history(history) + [{"role": "user", "content": message + note}])
        cards: list = []
        text = ""
        for _ in range(MAX_TOOL_ROUNDS):
            data = self.client.chat({"model": self.model, "messages": messages, "tools": OA_TOOLS,
                                     "temperature": 0.3, "max_tokens": 1024})
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                text = (msg.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for tc in calls:
                fn = tc.get("function", {})
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    out, _err = self._run_tool(fn.get("name", ""), args, cards)
                except json.JSONDecodeError:
                    out = {"error": "arguments were not valid JSON"}
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(out, default=str)})
        if not text:
            text = "Sorry, I had trouble answering that. Could you rephrase?"

        ctx = dict(context or {})
        suggestions = ["Where should I go this weekend?", "Best time to visit Netarhat"]
        for c in reversed(cards):
            d = c["data"]
            dest = d.get("destination") if isinstance(d.get("destination"), str) else None
            if c["type"] == "recommendations":
                ctx = {"destination": d["items"][0]["destination"], "date": d["date"]}
                suggestions = [f"Tell me more about {d['items'][0]['destination']}", "Show me a quieter option"]
            elif dest:
                ctx = {"destination": dest, **({"date": d["date"]} if isinstance(d.get("date"), str) else {})}
                suggestions = [f"7-day forecast for {dest}", f"Best time to visit {dest}", "Where should I go this weekend?"]
            break
        return {"reply": text, "cards": cards, "suggestions": suggestions, "context": ctx}
