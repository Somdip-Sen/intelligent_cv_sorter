import itertools

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

    def _slug(self, name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "", name.lower())
        if not s:
            s = "user"
        return s

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
            middle_pool = _flatten_list_maybe_hyphen(self.data["middle_names"])
            last_pool = _flatten_list_maybe_hyphen(self.data["last_names"])
            first = self.rng.choice(first_pool).split()[0]
            middle = self.rng.choice(middle_pool).split()[0]
            last = self.rng.choice(last_pool).split()[0]
            if random.random() < 0.9:
                # This type of name has a 90% chance of being executed
                name = f"{first} {last}"
            else:
                # This block has a 10% chance of being executed
                if middle is None:
                    # Fallback if middle name is not available
                    name = f"{first} {last}"
                else:
                    name = f"{first}{middle}{last}"
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
    def choose_email_domain(self) -> str:
        # Prefer gmail/outlook > others; fall back to YAML list if present
        pool = self.data.get("email_domains") or ["gmail.com", "outlook.com", "yahoo.co.in", "proton.me", "zoho.in"]
        # weights (sum to 1.0). Align by names if present; otherwise uniform.
        fixed_pref = {
            "gmail.com": 0.75,
            "outlook.com": 0.15
        }
        fixed_domain_keys = set(fixed_pref.keys())
        other_domains = sorted(
            [d for d in set(pool) if d not in fixed_domain_keys])  # Use set(pool) to handle duplicates
        remaining_prob = 1.0 - sum(fixed_pref.values())
        prob_per_other = 0
        final_domains = list(fixed_pref.keys()) + other_domains
        final_weights = list(fixed_pref.values()) + [prob_per_other] * len(other_domains)
        r = self.rng.random()
        cumulative_weights = itertools.accumulate(final_weights)
        for domain, cum_weight in zip(final_domains, cumulative_weights):
            # The running sum ensures we correctly map the random float `r` to a bucket.
            if r <= cum_weight:
                return domain
            # This fallback handles potential floating-point inaccuracies or an empty pool.
        return pool[-1] if pool else "gmail.com"

    def sample_email(self, name: str):
        base = name.lower().replace(" ", ".")
        dom = self.choose_email_domain()
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
    def sample_company(self):
        self.rng.choice(self.data["companies_curated"])

        # ----- salary (LPA) -----
    def sample_salary_lpa(self, seniority: str = "mid"):
        lo, hi = self.data["salary_lpa_by_seniority"].get(seniority, self.data["salary_lpa_by_seniority"]["mid"])
        low = int(lo + 0.1 * (hi - lo) * self.rng.random())
        high = int(hi - 0.1 * (hi - lo) * self.rng.random())
        if high <= low:
            high = low + 1
        return low, high

        # ----- college -----
    def sample_college(self):
        # flatten tiered lists gracefully
        tiers = []
        for k in ("colleges_tier1", "colleges_tier2", "colleges_tier3"):
            tiers.extend(self.data.get(k) or [])
        if not tiers:
            tiers = ["IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Guwahati", "IISc Bengaluru", "NIT Trichy",
                     "BITS Pilani"]
        return self.rng.choice(tiers)

    # ----- state -----
    def sample_state(self, city: str | None = None):
        # prefer explicit mapping if provided in YAML; else lightweight defaults to random state
        m = self.data.get("city_to_state") or {}
        states = self.data.get("states") or [
            "Karnataka", "Maharashtra", "Delhi", "Tamil Nadu", "Telangana", "Uttar Pradesh",
            "Gujarat", "West Bengal", "Rajasthan", "Haryana", "Punjab", "Kerala"
        ]
        if city and city in m:
            return m[city]
        return self.rng.choice(states)

        # ----- pin -----
    def sample_pin(self, city: str | None = None, state: str | None = None):
        by_city = self.data.get("pins_by_city") or {}
        if city and city in by_city:
            return str(self.rng.choice(by_city[city]))
        # simple synthetic fallback
        return str(self.rng.randint(100000, 799999))

        # --------github ------------
    def github_handle(self, name: str) -> str:
        base = self._slug(name)
        return f"{base}{self.rng.randint(11, 99)}"

    def sample_github(self, name: str) -> str:
        return f"https://github.com/{self.github_handle(name)}"

    # ----- LinkedIn -----
    def sample_linkedin(self, name: str):
        slug = self._slug(name)
        return f"https://linkedin.com/in/{slug}{self.rng.randint(11, 99)}"

    def sample_portfolio(self) -> str:
        # toy portfolio generator; could source from YAML if provided
        stem = self._slug(self.rng.choice(self.data.get("invent_syllables_last") or ["tech", "data", "code", "cloud"]))
        tld = self.rng.choice(["in", "dev", "io", "app"])
        return f"https://www.{stem}{self.rng.randint(11, 99)}.{tld}"

    # ----- sanitize -----
    def strip_non_indian_tokens(self, s: str):
        # DO NOT strip to ASCII; keep ₹ and friends.
        if not s:
            return s
        s = self._ascii(s)
        for t in ("USD", "$", "401k", "401(k)", "Social Security", "ZIP", "EEO (US)"):
            s = s.replace(t, "")
        return s
