"""Helpers for "Python Literacy for Text-as-Data".

Everyone pulls their own copy of the Manifesto corpus with their own free API
key. The Manifesto Project terms forbid redistributing the data, so the
workshop ships code and each participant pulls.

Get a key at https://manifesto-project.wzb.eu/signup (free, instant) and store
it in Colab secrets as MANIFESTO_KEY.
"""

import re
import time

import pandas as pd
import requests

MP_API = "https://manifesto-project.wzb.eu/api/v1"

# Pinned so every participant gets the same numbers on the day.
CORE_VERSION = "MPDS2026a"
CORPUS_VERSION = "2026-1"

# English-language countries, so the live session stays in one language.
DEFAULT_COUNTRIES = ("United Kingdom", "Ireland", "Australia", "New Zealand")
DEFAULT_YEAR_FROM = 2000

# The standard RILE code sets (Laver and Budge). The API returns numeric
# cmp_codes, so we match on codes and keep the names for reading.
RIGHT_CODES = {
    104: "Military: Positive",
    201: "Freedom and Human Rights",
    203: "Constitutionalism: Positive",
    305: "Political Authority",
    401: "Free Market Economy",
    402: "Incentives: Positive",
    407: "Protectionism: Negative",
    414: "Economic Orthodoxy",
    505: "Welfare State Limitation",
    601: "National Way of Life: Positive",
    603: "Traditional Morality: Positive",
    605: "Law and Order: Positive",
    606: "Civic Mindedness: Positive",
}
LEFT_CODES = {
    103: "Anti-Imperialism",
    105: "Military: Negative",
    106: "Peace",
    107: "Internationalism: Positive",
    202: "Democracy",
    403: "Market Regulation",
    404: "Economic Planning",
    406: "Protectionism: Positive",
    412: "Controlled Economy",
    413: "Nationalisation",
    504: "Welfare State Expansion",
    506: "Education Expansion",
    701: "Labour Groups: Positive",
}


# Handbook v5 splits some categories into subcategories like 605.1. They roll
# up to their three-digit parent, except these three, which point the opposite
# way from their parent and so count as uncoded. This is manifestoR's
# recode_v5_to_v4 rule.
V5_REVERSED = {"202.2", "605.2", "703.2"}


def collapse_rile(code):
    """Fold a cmp_code into 'left', 'right', or None for everything else."""
    code = str(code).strip()
    if code in V5_REVERSED:
        return None
    try:
        code = int(float(code))          # 605.1 rolls up to 605
    except (TypeError, ValueError):      # 'NA', 'H', blanks
        return None
    if code in LEFT_CODES:
        return "left"
    if code in RIGHT_CODES:
        return "right"
    return None


def _selfcheck():
    assert collapse_rile(504) == "left"
    assert collapse_rile("504") == "left"
    assert collapse_rile(605) == "right"
    assert collapse_rile("605.1") == "right"      # subcategory rolls up
    assert collapse_rile("605.2") is None         # reverses its parent
    assert collapse_rile("202.2") is None
    assert collapse_rile("202.1") == "left"
    assert collapse_rile("NA") is None
    assert collapse_rile("H") is None
    assert collapse_rile(None) is None
    assert collapse_rile(999) is None
    assert len(LEFT_CODES) == len(RIGHT_CODES) == 13
    assert not (set(LEFT_CODES) & set(RIGHT_CODES))
    print("collapse_rile checks pass")



# Party names the Manifesto Project stores under an English translation, so
# the native name survives in the text with nothing in the metadata to match
# it. Add to this if you swap in another country.
NATIVE_PARTY_TOKENS = {
    "gael", "fianna", "sinn", "fein", "aontu", "tories", "tory", "snp",
}


def manifesto_stopwords(df, extra=()):
    """English stopwords plus the country and party names in this corpus.

    Without these, LDA spends topics on who wrote a document rather than what
    it is about. The list is derived from the corpus so it follows the data.
    Substantive terms like 'maori' and institutions like 'westminster' stay in,
    since those name real policy domains.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    names = pd.concat([df["countryname"], df["partyname"],
                       df["partyabbrev"].astype(str)]).dropna().unique()
    tokens = set()
    for name in names:
        tokens |= set(re.findall(r"[a-zA-Z]{3,}", str(name).lower()))
    tokens |= NATIVE_PARTY_TOKENS
    tokens |= {"australian", "irish", "british", "english", "scottish", "welsh",
               "aotearoa", "ireland", "britain", "party", "parties"}
    return sorted(set(ENGLISH_STOP_WORDS) | tokens | set(extra))


def _get(endpoint, api_key, tries=4, **params):
    """One API call, retried. The server drops connections under load, and a
    dropped connection in the middle of a live session is not recoverable by
    hand, so retry rather than raise."""
    params["api_key"] = api_key
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(f"{MP_API}/{endpoint}", params=params, timeout=300)
        except requests.RequestException as e:
            # Scrub the key, it sits in the request URL and would land in output.
            last = str(e).replace(api_key, "<YOUR KEY>")
        else:
            if r.status_code == 401:
                raise RuntimeError(
                    "The Manifesto API rejected your key. Check that MANIFESTO_KEY "
                    "in Colab secrets matches the key on your profile page."
                )
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        if attempt < tries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Manifesto API failed on {endpoint} after {tries} tries. {last}")


def _core(api_key):
    """Main dataset as a DataFrame. The API sends the column names as row one."""
    payload = _get("get_core", api_key, key=CORE_VERSION)
    rows = payload if isinstance(payload, list) else next(
        v for v in payload.values() if isinstance(v, list)
    )
    core = pd.DataFrame(rows[1:], columns=rows[0])
    core["date"] = pd.to_numeric(core["date"], errors="coerce")
    core["party"] = pd.to_numeric(core["party"], errors="coerce")
    core["year"] = core["date"] // 100
    return core.dropna(subset=["date", "party"])


def pull_manifesto(api_key, countries=DEFAULT_COUNTRIES, year_from=DEFAULT_YEAR_FROM,
                   n_elections=2, parties_per_election=4, batch_size=15, verbose=True):
    """Pull annotated quasi-sentences into a tidy DataFrame.

    Returns columns text, cmp_code, rile, party, partyname, countryname, year.
    Downloading runs at roughly one manifesto per second on a cold cache, so
    n_elections and parties_per_election are what keep this inside a live session.
    """
    core = _core(api_key)
    picked = core[core["countryname"].isin(countries) & (core["year"] >= year_from)]
    if picked.empty:
        raise RuntimeError("No elections matched. Widen countries or year_from.")

    keys = [f"{int(p)}_{int(d)}" for p, d in zip(picked["party"], picked["date"])]
    meta = pd.DataFrame(
        _get("metadata", api_key, **{"keys[]": keys, "version": CORPUS_VERSION})["items"]
    )
    # The metadata endpoint names these differently from the main dataset.
    meta = meta.rename(columns={"party_id": "party", "election_date": "date"})
    # Only coded documents carry quasi-sentences.
    meta = meta[meta["annotations"] == True].dropna(subset=["manifesto_id"])
    if meta.empty:
        raise RuntimeError("No annotated documents in that selection.")
    meta = meta.astype({"party": int, "date": int}).merge(
        picked[["party", "date", "countryname", "partyname", "partyabbrev", "pervote"]]
        .astype({"party": int, "date": int}),
        on=["party", "date"], how="left",
    )
    # Coalition partners can file the same document under several party ids.
    # Identical text would double-count in the topic model and the training set.
    meta = meta.drop_duplicates(subset="md5sum_text")

    # The most recent elections per country, so there is year variation for the
    # STM covariate and for the Day 3 scaling, then the biggest parties in each,
    # which are the ones participants recognise.
    recent = (meta.groupby("countryname")["date"]
                  .apply(lambda s: sorted(s.unique())[-n_elections:]))
    meta = meta[[d in recent[c] for c, d in zip(meta["countryname"], meta["date"])]]
    meta["pervote"] = pd.to_numeric(meta["pervote"], errors="coerce").fillna(0)
    meta = (meta.sort_values("pervote", ascending=False)
                .groupby(["countryname", "date"], group_keys=False)
                .head(parties_per_election))
    if verbose:
        print(f"downloading {len(meta)} annotated manifestos")

    ids = meta["manifesto_id"].tolist()
    frames = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        payload = _get("texts_and_annotations", api_key,
                       **{"keys[]": chunk, "version": CORPUS_VERSION})
        for doc in payload["items"]:
            sentences = pd.DataFrame(doc["items"])
            if sentences.empty:
                continue
            sentences["manifesto_id"] = doc["key"]
            frames.append(sentences)
        if verbose:
            print(f"  {min(start + batch_size, len(ids))}/{len(ids)}")
    df = pd.concat(frames, ignore_index=True)

    # Attach party and year so Day 3 can aggregate to a left-right scale.
    df = df.merge(meta[["manifesto_id", "party", "date", "countryname", "partyname", "partyabbrev"]],
                  on="manifesto_id", how="left")
    df["year"] = df["date"] // 100
    df["rile"] = df["cmp_code"].map(collapse_rile)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 15].reset_index(drop=True)

    if verbose:
        print(f"pulled {len(df):,} quasi-sentences, "
              f"{df['rile'].notna().sum():,} of them left or right")
    return df


if __name__ == "__main__":
    _selfcheck()
