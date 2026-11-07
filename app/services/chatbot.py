"""Rule-based tourism chatbot.

No external AI service needed: it understands a handful of intents (predict, forecast,
recommend, best-time, list) using keyword + date parsing, then calls the ML predictor and
returns BOTH a short text reply and structured "cards" that the chat UI renders.

Stateless: the client echoes back `context` (last destination/date) with each message.
"""
import re
from datetime import date, timedelta

from app.services.predictor import FootfallPredictor
from ml.config import DESTINATIONS

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
MONTH_RE = "|".join(MONTHS)

DEST_ALIASES = {
    "dassam": "Dassam Falls", "hundru": "Hundru Falls", "netarhat": "Netarhat",
    "deoghar": "Deoghar", "baidyanath": "Deoghar", "betla": "Betla National Park",
    "patratu": "Patratu Valley",
}
CATEGORY_WORDS = {
    "waterfall": ["waterfall", "falls", "fall "],
    "hill_station": ["hill", "mountain", "sunset", "sunrise"],
    "religious": ["temple", "religious", "pilgrim", "spiritual", "dham"],
    "wildlife": ["wildlife", "safari", "animal", "tiger", "elephant", "national park", "forest"],
    "scenic": ["scenic", "lake", "valley", "dam", "view", "nature"],
}
LEVEL_TEXT = {"LOW": "low", "MODERATE": "moderate", "HIGH": "high", "VERY_HIGH": "very high"}


def _fmt(d: date) -> str:
    return d.strftime("%a, %d %b %Y")


def _next_weekday(today: date, wd: int, force_next=False) -> date:
    delta = (wd - today.weekday()) % 7
    if delta == 0 and force_next:
        delta = 7
    return today + timedelta(days=delta)


def parse_date(text: str, today: date) -> date | None:
    t = text.lower()
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", t)
    if m:
        day, mon = int(m[1]), int(m[2])
        yr = int(m[3]) if m[3] else today.year
        yr += 2000 if yr < 100 else 0
        try:
            d = date(yr, mon, day)
            return d if m[3] or d >= today else d.replace(year=yr + 1)
        except ValueError:
            pass
    for pat, day_g, mon_g in ((rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({MONTH_RE})[a-z]*\b", 1, 2),
                              (rf"\b({MONTH_RE})[a-z]*\s+(\d{{1,2}})(?!\d)", 2, 1)):
        m = re.search(pat, t)
        if m:
            try:
                d = date(today.year, MONTHS[m[mon_g][:3]], int(m[day_g]))
                return d if d >= today else d.replace(year=today.year + 1)
            except ValueError:
                pass
    if "day after tomorrow" in t:
        return today + timedelta(days=2)
    if "tomorrow" in t:
        return today + timedelta(days=1)
    if re.search(r"\btoday\b|\btonight\b", t):
        return today
    if "next weekend" in t:
        return _next_weekday(today, 5, True) + timedelta(days=7 if today.weekday() < 5 else 0)
    if "weekend" in t:
        return today if today.weekday() >= 5 else _next_weekday(today, 5)
    m = re.search(r"\b(next\s+)?(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b", t)
    if m and not re.search(r"\b(sun|mon)[a-z]*\s*(rise|set)", t):
        return _next_weekday(today, WEEKDAYS[m[2]], force_next=bool(m[1]))
    return None


def find_destination(text: str) -> str | None:
    t = text.lower()
    for alias, name in DEST_ALIASES.items():
        if alias in t:
            return name
    return None


def find_categories(text: str) -> list[str]:
    t = text.lower() + " "
    return [c for c, words in CATEGORY_WORDS.items() if any(w in t for w in words)]


class TourismChatbot:
    def __init__(self, predictor: FootfallPredictor):
        self.p = predictor

    # ---- response builders ----------------------------------------------
    @staticmethod
    def _out(reply, cards=None, suggestions=None, context=None):
        return {"reply": reply, "cards": cards or [], "suggestions": suggestions or [], "context": context or {}}

    def _default_suggestions(self):
        return ["Where should I go this weekend?", "How crowded is Dassam Falls tomorrow?",
                "Best time to visit Netarhat", "Show all destinations"]

    def _predict(self, dest, d):
        r = self.p.predict(dest, d)
        reply = (f"On {_fmt(d)}, {dest} is expected to have {LEVEL_TEXT[r['crowd_level']]} crowds "
                 f"(about {r['predicted_visitors']:,} visitors).")
        if r["crowd_level"] in ("HIGH", "VERY_HIGH"):
            reply += " If you prefer a quieter trip, I can suggest alternatives."
        sugg = [f"7-day forecast for {dest}", f"Best time to visit {dest}", f"Alternatives on {d.strftime('%d %b')}"]
        return self._out(reply, [{"type": "prediction", "data": {**r, "date": r["date"].isoformat()}}],
                         sugg, {"destination": dest, "date": d.isoformat()})

    def _forecast(self, dest, start):
        f = self.p.forecast(dest, start, 7)
        days = [{**x, "date": x["date"].isoformat()} for x in f["days"]]
        best = min(f["days"], key=lambda x: x["predicted_visitors"])
        busiest = max(f["days"], key=lambda x: x["predicted_visitors"])
        reply = (f"Here is the 7-day outlook for {dest} from {_fmt(start)}. "
                 f"Quietest day: {_fmt(best['date'])}; busiest: {_fmt(busiest['date'])}.")
        return self._out(reply, [{"type": "forecast", "data": {"destination": dest, "days": days,
                                  "change_pct": f["change_vs_previous_period_pct"]}}],
                         [f"Best time to visit {dest}", "Where should I go this weekend?"],
                         {"destination": dest, "date": start.isoformat()})

    def _monthly(self, dest, today):
        prof = self.p.monthly_profile(dest, today.year)
        reply = (f"{dest} is busiest around {prof['peak_month']} and quietest around {prof['quietest_month']} "
                 "(typical daily visitors, based on the model).")
        return self._out(reply, [{"type": "monthly", "data": prof}],
                         [f"7-day forecast for {dest}", "Where should I go this weekend?"],
                         {"destination": dest})

    def _recommend(self, d, prefs, note=""):
        r = self.p.recommend(d, prefs, True, 3)
        recs = r["recommendations"]
        top = recs[0]
        what = f" matching {', '.join(p.replace('_', ' ') for p in prefs)}" if prefs else ""
        reply = (f"{note}For {_fmt(d)}{what}, I'd suggest {top['destination']}: "
                 f"{LEVEL_TEXT[top['crowd_level']]} predicted crowds.")
        return self._out(reply, [{"type": "recommendations", "data": {"date": d.isoformat(), "items": recs}}],
                         [f"Tell me more about {top['destination']}", "Try another date"],
                         {"destination": top["destination"], "date": d.isoformat()})

    # ---- main entry ------------------------------------------------------
    def respond(self, message: str, context: dict | None = None, today: date | None = None) -> dict:
        today = today or date.today()
        ctx = context or {}
        t = message.lower().strip()
        if not t:
            return self._out("Type a question about a destination or date.", suggestions=self._default_suggestions())

        dest = find_destination(t)
        d = parse_date(t, today)
        prefs = find_categories(t)
        ctx_dest = ctx.get("destination")
        ctx_date = date.fromisoformat(ctx["date"]) if ctx.get("date") else None

        words = set(re.findall(r"[a-z]+", t))
        if not dest and not d and not prefs:
            if words & {"hi", "hello", "hey", "namaste", "hii", "hola"}:
                return self._out("Namaste! 🙏 I'm your Jharkhand travel assistant. I can predict how crowded a "
                                 "place will be on a given day and suggest the best spots for you.",
                                 suggestions=self._default_suggestions())
            if words & {"thanks", "thank", "thx", "bye", "goodbye"}:
                return self._out("You're welcome! Enjoy your trip to Jharkhand. 🌿", suggestions=self._default_suggestions())
            if "help" in words or "what can you do" in t:
                return self._out("Ask me things like: 'How crowded is Hundru Falls on 15 August?', "
                                 "'Where should I go this weekend?' or 'Best time to visit Netarhat'.",
                                 suggestions=self._default_suggestions())
            if re.search(r"destination|places|list|all spots", t):
                names = ", ".join(DESTINATIONS)
                return self._out(f"I cover: {names}. Ask me about any of them.",
                                 suggestions=[f"How crowded is {n} this weekend?" for n in list(DESTINATIONS)[:3]])
            if "alternative" in t and ctx_date:
                return self._recommend(ctx_date, [])
            if re.search(r"try another date|another date", t):
                return self._out("Sure, which date? For example: 'tomorrow', 'this weekend' or '15 August'.",
                                 context=ctx)

        target = dest or ctx_dest
        if re.search(r"best time|which month|best month|season|peak|when (should|to)", t) and target:
            return self._monthly(target, today)
        if re.search(r"forecast|\bnext week\b|\bthis week\b|week ahead|trend|7[- ]day", t) and target:
            start = _next_weekday(today, 0, True) if re.search(r"\bnext week\b", t) else (d or today)
            return self._forecast(target, start)
        if re.search(r"alternative|instead|quieter|less crowd|other place", t) and (d or ctx_date):
            return self._recommend(d or ctx_date, prefs)

        wants_reco = bool(re.search(r"where|recommend|suggest|plan|best place|should i|go |visit|trip", t)) or bool(prefs)
        if not dest and wants_reco:
            note = "" if d else "No date given, so I used this weekend. "
            return self._recommend(d or ctx_date or parse_date("this weekend", today), prefs, note)

        if dest:
            when = d or ctx_date
            if when:
                return self._predict(dest, when)
            return self._out(f"Which date are you planning for {dest}? (e.g. 'tomorrow', 'this weekend', '15 August')",
                             suggestions=["Tomorrow", "This weekend", "15 August"], context={"destination": dest})
        if d:
            if ctx_dest:
                return self._predict(ctx_dest, d)
            return self._recommend(d, prefs)

        return self._out("I didn't quite get that. I can predict crowds for a destination and date, "
                         "or suggest where to go. Try one of these:", suggestions=self._default_suggestions(), context=ctx)
