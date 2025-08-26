import yaml, random, threading, re
from pathlib import Path


def _flatten_list_maybe_hyphen(rows):
    """
    Many entries in your YAML use ' - ' inside a single item (not commas).
    This flattens lists like ["A - B - C"] into ["A","B","C"].
    """
    if not rows: return []
    out = []
    for row in rows:
        if isinstance(row, str) and " - " in row:
            out.extend([x.strip() for x in row.split(" - ") if x.strip()])
        else:
            out.append(row)
    return out


# Lightweight maps for state/pin inference for realism.
_CITY_TO_STATE = {
    # core
    "Ahmedabad": "Gujarat",
    "Bengaluru": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Gurgaon": "Haryana",
    "Hyderabad": "Telangana",
    "Kolkata": "West Bengal",
    "Mumbai": "Maharashtra",
    "New Delhi": "Delhi",
    "Noida": "Uttar Pradesh",
    "Pune": "Maharashtra",
    # a few common extras
    "Jaipur": "Rajasthan",
    "Lucknow": "Uttar Pradesh",
    "Indore": "Madhya Pradesh",
    "Chandigarh": "Chandigarh",
    "Nagpur": "Maharashtra",
    "Coimbatore": "Tamil Nadu",
    "Surat": "Gujarat",
    "Varanasi": "Uttar Pradesh",
    "Visakhapatnam": "Andhra Pradesh",
    "Bhopal": "Madhya Pradesh",
    "Vadodara": "Gujarat",
    "Mysore": "Karnataka",
    "Thane": "Maharashtra",
    "Kochi": "Kerala",
    "Patna": "Bihar",
    "Ranchi": "Jharkhand",
}


# Typical city/state PIN prefixes (rough, not exhaustive)
_STATE_PIN_PREFIX = {
    "Delhi": [110],
    "Maharashtra": [400, 401, 402, 403, 411, 412],
    "Karnataka": [560, 561, 562],
    "Tamil Nadu": [600, 601, 602],
    "Telangana": [500, 501, 502],
    "Gujarat": [380, 382, 390, 395],
    "West Bengal": [700, 701],
    "Uttar Pradesh": [201, 202, 226, 247, 281],
    "Rajasthan": [302, 303, 311],
    "Madhya Pradesh": [452, 462, 474],
    "Andhra Pradesh": [520, 530, 531],
    "Kerala": [670, 680, 682],
    "Bihar": [800, 801],
    "Haryana": [120, 121, 122, 124],
    "Jharkhand": [834, 829],
    "Chandigarh": [160],
}

_COMMON_STATES = list(_STATE_PIN_PREFIX.keys())


class IndiaBank:
    def __init__(self, path: str | Path, p_invent: float = 0.15, seed: int = 42):
        self.data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.rng = random.Random(seed)
        self.p_inv = max(0.0, min(1.0, float(p_invent)))
        self._lock = threading.Lock()
        self._used = {"names": set(), "phones": set(), "emails": set()}

        # Normalize possibly hyphen-joined rows in YAML
        for k in ("first_names", "middel_names", "last_names", "cities_core",
                  "cities_extra", "companies_curated", "email_domains", "colleges"):
            if k in self.data:
                self.data[k] = _flatten_list_maybe_hyphen(self.data[k])

    # ----- helpers -----
    def _coin(self):
        return self.rng.random() < self.p_inv

    def _ascii(self, s: str) -> str:
        return s.encode("ascii", "ignore").decode("ascii")

    # ----- name -----
    def _invent_name(self):
        f = "".join(self.rng.choices(self.data["invent_syllables_first"], k=self.rng.randint(1, 2)))
        l = "".join(self.rng.choices(self.data["invent_syllables_last"], k=self.rng.randint(1, 2)))
        # normalize caps
        f = f.capitalize()
        l = l.capitalize()
        return f"{f} {l}"

    def sample_name(self):
        if self._coin():
            name = self._invent_name()
        else:
            first_pool = _flatten_list_maybe_hyphen(self.data["first_names"])
            last_pool = _flatten_list_maybe_hyphen(self.data["last_names"])
            first = self.rng.choice(first_pool).split()[0]
            last = self.rng.choice(last_pool).split()[0]
            name = f"{first} {last}"
        with self._lock:
            while name in self._used["names"]:
                name = self._invent_name()
            self._used["names"].add(name)
        return name

        # ----- phone -----

    def sample_phone(self):
        def mk():
            start = self.rng.choice(["6", "7", "8", "9"])
            return "+91" + start + "".join(str(self.rng.randint(0, 9)) for _ in range(9))

        with self._lock:
            ph = mk()
            while ph in self._used["phones"]:
                ph = mk()
            self._used["phones"].add(ph)
        return ph

        # ----- email -----

    def _weighted_domain(self):
        domains = _flatten_list_maybe_hyphen(self.data.get("email_domains", []))
        if not domains:
            return "gmail.com"
        # 70% gmail/outlook, 30% others
        if self.rng.random() < 0.7:
            for pref in ("gmail.com", "outlook.com"):
                if pref in domains:
                    return pref
            # fallback if not present
        return self.rng.choice(domains)

    def sample_email(self, name: str):
        base = name.lower().replace(" ", ".")
        # bias towards gmail/outlook
        domains = self.data.get("email_domains") or ["gmail.com", "outlook.com", "yahoo.co.in", "proton.me", "zoho.in"]
        weighted = (["gmail.com"] * 5) + (["outlook.com"] * 3) + [d for d in domains if
                                                                  d not in {"gmail.com", "outlook.com"}]
        dom = self.rng.choice(weighted)
        with self._lock:
            em = f"{base}{self.rng.randint(11, 99)}@{dom}"
            while em in self._used["emails"]:
                em = f"{base}{self.rng.randint(11, 99)}@{dom}"
            self._used["emails"].add(em)
        return em

        # ----- city -----

    def sample_city(self):
        # More frequency on core cities (extras only if _coin() triggers)
        pool = self.data["cities_extra"] if self._coin() else self.data["cities_core"]
        return self.rng.choice(pool)

        # ----- company -----

    def _invent_company(self):
        stem = self.rng.choice(self.data["invent_syllables_last"]).capitalize()
        suf = self.rng.choice(self.data["company_suffixes"])
        return f"{stem} {suf}"

    def sample_company(self):
        if self._coin():
            return self._invent_company()
        return self.rng.choice(self.data["companies_curated"])

        # ----- salary (LPA) -----

    def sample_salary_lpa(self, seniority: str = "mid"):
        lo, hi = self.data["salary_lpa_by_seniority"].get(seniority, self.data["salary_lpa_by_seniority"]["mid"])
        low = int(lo + 0.1 * (hi - lo) * self.rng.random())
        high = int(hi - 0.1 * (hi - lo) * self.rng.random())
        if high <= low: high = low + 1
        return low, high

        # ----- college -----

    def sample_college(self):
        pools = []
        for k in ("colleges_tier1", "colleges_core", "colleges_tier2", "colleges_extra"):
            if isinstance(self.data.get(k), list):
                pools.extend(self.data[k])
        if not pools:
            pools = ["IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Guwahati", "IISc Bengaluru", "NIT Trichy",
                     "BITS Pilani"]
        return self.rng.choice(pools)

    # ----- state -----
    def sample_state(self, city: str | None = None):
        # prefer explicit mapping if provided in YAML; else random state
        def sample_state(self, city: str | None = None):
            m = self.data.get("state_by_city") or {}
            if city and city in m:
                return m[city]
            states = self.data.get("states") or [
                "Karnataka", "Maharashtra", "Tamil Nadu", "Telangana", "Delhi", "Haryana",
                "Uttar Pradesh", "Gujarat", "West Bengal", "Rajasthan", "Kerala", "Madhya Pradesh"
            ]
            return self.rng.choice(states)

        # ----- pin -----

    def sample_pin(self, city: str | None = None, state: str | None = None):
        pins_by_state = self.data.get("pins_by_state") or {}
        if state and state in pins_by_state and pins_by_state[state]:
            return str(self.rng.choice(pins_by_state[state]))
        # plausible fallback: 110xxx / 400xxx / 560xxx etc
        head = self.rng.choice([110, 400, 500, 560, 600, 700, 800])
        tail = self.rng.randint(100, 999)
        return f"{head}{tail:03d}"

    # ----- LinkedIn -----
    def sample_linkedin(self, name: str):
        slug = self._ascii(name.lower().replace(" ", "-"))
        return f"linkedin.com/in/{slug}{self.rng.randint(10, 99)}"

        # ----- sanitize -----

    def strip_non_indian_tokens(self, s: str):
        # DO NOT strip to ASCII; keep ₹ and friends.
        if not s:
            return s
        for t in ("USD", "$", "401k", "401(k)", "Social Security", "ZIP", "EEO (US)"):
            s = s.replace(t, "")
        # Normalize spacing that earlier ASCII-pass used to disturb
        s = re.sub(r"\s{2,}", " ", s)
        return s
