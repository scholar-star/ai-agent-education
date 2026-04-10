"""
seed.yaml v2.0 — 입문자 신디 추천
- Streamlit 하이브리드 UI (위저드 + 채팅)
- LangGraph: 예산 우선 router → 조건부 엣지 → ReAct(create_react_agent)
- 로컬 목 + Discogs 검색/릴리즈 상세 + Wikidata 엔티티 클레임 (공식 HTTP API만)
- explain_beginner: 위키백과 + dictionaryapi.dev
- RAG: text-embedding-3-small + 가이드 MD + 로컬 카탈로그 요약(추천 맥락 검색)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Wikipedia / 사전 API 호출 시 식별 가능한 UA (https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy)
_WIKI_UA = "SynthRecommenderApp/1.0 (https://github.com/education-local; ko+en Wikipedia API)"

# Discogs: https://www.discogs.com/developers — UA 필수, Personal Token 선택(분당 한도 증가)
_DISCOGS_UA = "SynthRecommenderApp/1.0 +https://github.com/education-local"

# Wikidata: https://www.wikidata.org/wiki/Wikidata:Data_access
_WIKIDATA_UA = "SynthRecommenderApp/1.0 (Wikidata API; educational use)"


def _http_get_json(url: str, timeout: float = 10.0) -> Any | None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _WIKI_UA,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _http_get_json_headers(url: str, headers: dict[str, str], timeout: float = 12.0) -> Any | None:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _discogs_headers() -> dict[str, str]:
    h = {"User-Agent": _DISCOGS_UA, "Accept": "application/json"}
    token = (os.getenv("DISCOGS_PERSONAL_TOKEN") or "").strip()
    if token:
        h["Authorization"] = f"Discogs token={token}"
    return h


def _discogs_parse_release_id(ref: str) -> str | None:
    s = (ref or "").strip()
    if not s:
        return None
    m = re.search(r"releases?[/\\](\d+)", s, re.I)
    if m:
        return m.group(1)
    if s.isdigit():
        return s
    return None


def discogs_fetch_release(release_id: str) -> dict[str, Any] | None:
    """GET https://api.discogs.com/releases/{id} — 문서화된 공개 API."""
    rid = _discogs_parse_release_id(release_id)
    if not rid:
        return None
    url = f"https://api.discogs.com/releases/{rid}"
    data = _http_get_json_headers(url, _discogs_headers(), timeout=15.0)
    if not isinstance(data, dict):
        return None
    artists = []
    for a in (data.get("artists") or [])[:8]:
        if isinstance(a, dict) and a.get("name"):
            artists.append(a["name"])
    formats_out: list[str] = []
    for f in (data.get("formats") or [])[:6]:
        if isinstance(f, dict):
            nm = f.get("name") or ""
            desc = ", ".join(f.get("descriptions") or []) if isinstance(f.get("descriptions"), list) else ""
            formats_out.append(f"{nm} ({desc})".strip(" ()") or nm)
    tracklist = data.get("tracklist") or []
    n_tracks = len(tracklist) if isinstance(tracklist, list) else 0
    ids = []
    for x in (data.get("identifiers") or [])[:12]:
        if isinstance(x, dict):
            ids.append({"type": x.get("type"), "value": x.get("value")})
    return {
        "source": "discogs_release_api",
        "release_id": int(rid) if rid.isdigit() else rid,
        "title": data.get("title"),
        "year": data.get("year"),
        "country": data.get("country"),
        "genres": data.get("genres") or [],
        "styles": data.get("styles") or [],
        "formats_summary": formats_out,
        "artists": artists,
        "track_count": n_tracks,
        "identifiers": ids,
        "uri": data.get("uri"),
        "resource_url": data.get("resource_url"),
        "notes_preview": ((data.get("notes") or "")[:400] + "…") if len((data.get("notes") or "")) > 400 else data.get("notes"),
        "disclaimer_ko": "참고로 이건 Discogs에 올라온 ‘음반/릴리즈’ 정보예요. 매장 가격이랑은 다를 수 있고요, 신디 스펙 시트는 아니에요.",
    }


def _wikidata_get_json(params: dict[str, str]) -> Any | None:
    qs = urllib.parse.urlencode(params)
    url = f"https://www.wikidata.org/w/api.php?{qs}"
    return _http_get_json_headers(
        url,
        {"User-Agent": _WIKIDATA_UA, "Accept": "application/json"},
        timeout=15.0,
    )


def _wikidata_snak_value(snak: Any) -> Any:
    if not isinstance(snak, dict):
        return None
    if snak.get("snaktype") == "novalue":
        return None
    if snak.get("snaktype") == "somevalue":
        return "somevalue"
    dv = snak.get("datavalue") or {}
    val = dv.get("value")
    dt = snak.get("datatype") or ""
    if dt == "wikibase-entityid" and isinstance(val, dict):
        return val.get("id")
    if dt == "quantity" and isinstance(val, dict):
        return val.get("amount")
    if dt == "time" and isinstance(val, dict):
        return val.get("time")
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, dict):
        return val.get("text") or str(val)[:240]
    return str(val)[:240] if val is not None else None


def wikidata_entity_specs_fetch(search_query: str, max_props: int = 18) -> dict[str, Any]:
    """
    wbsearchentities → 첫 매치에 대해 wbgetentities (claims 일부).
    기기/소프트에 자주 쓰이는 속성을 우선 나열하고, 나머지는 개수 제한.
    """
    q = (search_query or "").strip()
    if not q:
        return {"source": "wikidata", "error": "empty_query", "claims": {}}

    sea = _wikidata_get_json(
        {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": q,
            "limit": "5",
        }
    )
    if not isinstance(sea, dict) or not (sea.get("search") or []):
        return {
            "source": "wikidata",
            "entity_id": None,
            "label_en": None,
            "note_ko": "Wikidata에서 못 찾았어요. 제품 영문 풀네임으로 한번 더 검색해 볼까요?",
            "claims": {},
        }

    first = sea["search"][0]
    qid = first.get("id")
    if not qid:
        return {"source": "wikidata", "error": "no_id", "claims": {}}

    ent = _wikidata_get_json(
        {
            "action": "wbgetentities",
            "format": "json",
            "ids": qid,
            "props": "labels|descriptions|claims",
            "languages": "en",
        }
    )
    if not isinstance(ent, dict):
        return {"source": "wikidata", "entity_id": qid, "error": "fetch_failed", "claims": {}}

    entities = ent.get("entities") or {}
    node = entities.get(qid)
    if not isinstance(node, dict):
        return {"source": "wikidata", "entity_id": qid, "error": "missing_entity", "claims": {}}

    label_en = (node.get("labels") or {}).get("en", {}).get("value")
    desc_en = (node.get("descriptions") or {}).get("en", {}).get("value")
    claims_raw = node.get("claims") or {}

    priority_props = [
        "P31",
        "P279",
        "P176",
        "P178",
        "P186",
        "P2079",
        "P366",
        "P495",
        "P571",
        "P1072",
        "P1552",
        "P2067",
        "P2043",
    ]
    out_claims: dict[str, list[Any]] = {}
    seen: set[str] = set()

    def add_prop(pid: str) -> None:
        nonlocal out_claims, seen
        if pid in seen or len(out_claims) >= max_props:
            return
        lst = claims_raw.get(pid)
        if not isinstance(lst, list):
            return
        vals: list[Any] = []
        for c in lst[:3]:
            if not isinstance(c, dict):
                continue
            m = c.get("mainsnak")
            v = _wikidata_snak_value(m)
            if v is not None:
                vals.append(v)
        if vals:
            out_claims[pid] = vals
            seen.add(pid)

    for pid in priority_props:
        add_prop(pid)
    for pid in sorted(claims_raw.keys()):
        if len(out_claims) >= max_props:
            break
        add_prop(pid)

    return {
        "source": "wikidata",
        "entity_id": qid,
        "label_en": label_en,
        "description_en": desc_en,
        "claims": out_claims,
        "property_note_ko": "속성 이름이 P31 같은 코드로 나와요(예: P31은 ‘무엇의 한 종류인지’). 값은 Q번호나 글자로 와요.",
        "disclaimer_ko": "여기 적힌 건 Wikidata에 실제로 있는 내용만이에요. 없는 스펙은 억지로 만들지 말아 주세요.",
    }


def _discogs_enabled() -> bool:
    return os.getenv("DISCOGS_SEARCH", "1").strip().lower() not in ("0", "false", "no", "off")


def _discogs_release_kind(formats: Any) -> str:
    """릴리즈 포맷 문자열로 hardware(실체 매체) vs software(파일·디지털) 휴리스틱."""
    if isinstance(formats, str):
        parts = [formats]
    elif isinstance(formats, list):
        parts = [str(x) for x in formats]
    else:
        parts = []
    blob = " ".join(parts).lower()
    if any(k in blob for k in ("file", "flac", "mp3", "wav", "alac", "digital")):
        return "software"
    return "hardware"


def _discogs_search_releases(query: str, per_page: int = 8) -> list[dict[str, Any]]:
    """
    Discogs /database/search (type=release).
    - 토큰 없이도 호출 가능(분당 25회). DISCOGS_PERSONAL_TOKEN 있으면 60회.
    - 결과는 '앨범/싱글 릴리즈'이지 쇼핑몰 SKU가 아님.
    """
    if not _discogs_enabled() or not (query or "").strip():
        return []
    q = f"{query.strip()} synthesizer"
    params = urllib.parse.urlencode(
        {"q": q, "type": "release", "per_page": str(per_page)}
    )
    url = f"https://api.discogs.com/database/search?{params}"
    headers = _discogs_headers()
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for r in (data.get("results") or [])[:per_page]:
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "").strip() or "(제목 없음)"
        fmt = r.get("format")
        kind = _discogs_release_kind(fmt)
        year = r.get("year") or ""
        country = r.get("country") or ""
        genre = ", ".join(r.get("genre") or []) if isinstance(r.get("genre"), list) else ""
        uri = r.get("uri") or ""
        web = f"https://www.discogs.com{uri}" if uri.startswith("/") else (uri or "")
        fmt_s = ", ".join(fmt) if isinstance(fmt, list) else str(fmt or "")
        rid = r.get("id")
        blurb = (
            f"Discogs에서 찾은 릴리즈예요 · 나온 해 {year} · 포맷은 {fmt_s}"
            + (f" · 나라 {country}" if country else "")
            + (f" · 장르 느낌 {genre}" if genre else "")
            + " · (가격·재고랑은 별개예요) · 더 알고 싶으면 discogs_release_id로 상세 조회해 보세요"
        )
        rows.append(
            {
                "name": title,
                "kind": kind,
                "bucket": None,
                "price_krw": None,
                "blurb": blurb,
                "source": "discogs",
                "web": web,
                "discogs_release_id": rid,
            }
        )
    return rows


def _wiki_opensearch_first_title(lang: str, term: str) -> str | None:
    """opensearch로 가장 유사한 문서 제목 1개."""
    q = urllib.parse.urlencode(
        {
            "action": "opensearch",
            "search": term,
            "limit": "5",
            "namespace": "0",
            "format": "json",
        }
    )
    url = f"https://{lang}.wikipedia.org/w/api.php?{q}"
    data = _http_get_json(url)
    if not isinstance(data, list) or len(data) < 2:
        return None
    titles = data[1]
    if not isinstance(titles, list) or not titles:
        return None
    t0 = titles[0]
    return str(t0).strip() if t0 else None


def _wiki_page_extract(lang: str, title: str) -> tuple[str | None, str | None]:
    """문서 첫 단락(plain)과 정규 URL."""
    qs = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        }
    )
    url = f"https://{lang}.wikipedia.org/w/api.php?{qs}"
    data = _http_get_json(url)
    if not isinstance(data, dict):
        return None, None
    pages = data.get("query", {}).get("pages") or {}
    extract: str | None = None
    canon_title = title
    for _pid, page in pages.items():
        if not isinstance(page, dict):
            continue
        if page.get("missing"):
            continue
        raw = (page.get("extract") or "").strip()
        if raw:
            extract = raw
            canon_title = page.get("title") or title
            break
    if not extract:
        return None, None
    slug = canon_title.replace(" ", "_")
    page_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(slug, safe='()%_')}"
    return extract, page_url


def _dictionaryapi_dev_english(word: str) -> str | None:
    """https://dictionaryapi.dev — 무료, 키 불필요 (영어 단어 위주)."""
    w = urllib.parse.quote(word.strip().lower(), safe="")
    if not w:
        return None
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{w}"
    data = _http_get_json(url)
    if not isinstance(data, list) or not data:
        return None
    bits: list[str] = []
    for meaning in (data[0].get("meanings") or [])[:3]:
        pos = meaning.get("partOfSpeech") or ""
        for d in (meaning.get("definitions") or [])[:2]:
            t = (d.get("definition") or "").strip()
            if t:
                bits.append(f"[{pos}] {t}" if pos else t)
    out = "\n".join(bits).strip()
    return out[:1500] if out else None


def _cosine_sim_vec(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)


_DEFAULT_RAG_CHUNKS: list[str] = [
    "MIDI는 노트와 컨트롤을 주고받는 규약이다. USB-MIDI로 PC와 신디를 연결한다.",
    "폴리포니는 동시 발음 수다. 코드에는 폴리, 베이스 리드에는 모노가 흔하다.",
    "VA는 아날로그를 디지털로 모델링한 신디 타입이다.",
]


class _SynthRAGIndex:
    """가게 안내집 MD + 진열 메모(후보 요약)를 임베딩으로 찾아요."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._n_guide_chunks: int = 0
        self._vectors: list[list[float]] | None = None
        self._embeddings: OpenAIEmbeddings | None = None
        self._lock = threading.Lock()

    def _ensure_index(self) -> str | None:
        with self._lock:
            if self._vectors is not None:
                return None
            if not os.getenv("OPENAI_API_KEY"):
                return "임베딩 검색하려면 OPENAI_API_KEY가 있어야 해요."
            self._chunks, self._n_guide_chunks = _build_rag_chunks_split()
            if not self._chunks:
                return "안내 자료(코퍼스)가 비어 있어요."
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self._vectors = self._embeddings.embed_documents(self._chunks)
        return None

    def search(self, query: str, k: int = 4) -> dict[str, Any]:
        err = self._ensure_index()
        if err:
            return {"source": "rag", "error": err}
        assert self._embeddings is not None and self._vectors is not None
        qv = self._embeddings.embed_query((query or "").strip())
        scored: list[tuple[float, int]] = []
        for i, doc_v in enumerate(self._vectors):
            scored.append((_cosine_sim_vec(qv, doc_v), i))
        scored.sort(key=lambda x: -x[0])
        hits: list[dict[str, Any]] = []
        for score, idx in scored[:k]:
            origin = "입문가이드" if idx < self._n_guide_chunks else "로컬카탈로그"
            hits.append(
                {
                    "chunk_id": idx,
                    "score": round(float(score), 4),
                    "origin": origin,
                    "text": self._chunks[idx][:2400],
                }
            )
        return {
            "source": "rag",
            "query": query,
            "hits": hits,
            "note_ko": "가게 안내 책자랑 진열된 모델 메모를 같이 뒤져봤어요. 최종 가격·후보는 꼭 search_catalog로 다시 확인해 주세요.",
        }


_rag_singleton: _SynthRAGIndex | None = None
_rag_singleton_lock = threading.Lock()


def _get_rag_index() -> _SynthRAGIndex:
    global _rag_singleton
    with _rag_singleton_lock:
        if _rag_singleton is None:
            _rag_singleton = _SynthRAGIndex()
        return _rag_singleton


def lookup_term_public_sources(term: str) -> str:
    """위키(ko→en)랑 영어 무료 사전으로 용어 풀어 드릴 때 씁니다."""
    raw = (term or "").strip()
    if not raw:
        return "어떤 말이 궁금하신지 한 번만 더 적어 주시겠어요?"

    for lang in ("ko", "en"):
        title = _wiki_opensearch_first_title(lang, raw) or raw
        extract, page_url = _wiki_page_extract(lang, title)
        if extract:
            body = extract[:2000].strip()
            if len(extract) > 2000:
                body += "…"
            src = f"\n\n출처: {page_url}" if page_url else ""
            return f"「{title}」 (위키백과 {lang})\n{body}{src}"

    # 한글 등 비라틴이 섞이지 않은 경우 영어 무료 사전 시도
    if raw.isascii():
        dfn = _dictionaryapi_dev_english(raw)
        if dfn:
            return (
                f"「{raw}」 (Free Dictionary API · 영어)\n{dfn}\n\n"
                "출처: https://dictionaryapi.dev/"
            )

    return (
        f"'{raw}'는 제가 쓰는 사전/API에서 금방 못 찾았어요. "
        "망이 끊겼거나 차단됐을 수도 있고요, 영문으로 쓰인 정식 이름이 있으면 그걸로 한번 더 물어봐 주세요."
    )


@tool
def explain_beginner(term: Annotated[str, "헷갈리는 용어"]) -> str:
    """손님이 헷갈리는 말을 풀어 드릴 때 씁니다. 위키·영어 사전 API를 봅니다."""
    return lookup_term_public_sources(term)


@tool
def get_discogs_release(
    release_ref: Annotated[
        str,
        "search_catalog에 나온 discogs_release_id 숫자, 또는 .../releases/123 주소",
    ],
) -> str:
    """Discogs에서 그 음반/릴리즈 자세한 정보를 더 가져올 때 써요. 매장 시세랑은 달라요."""
    if not _discogs_enabled():
        return json.dumps(
            {"error": "지금은 Discogs 검색이 꺼져 있어요", "hint": "DISCOGS_SEARCH를 0이 아니게 켜 주세요"},
            ensure_ascii=False,
        )
    data = discogs_fetch_release(release_ref)
    if not data:
        return json.dumps(
            {
                "error": "이 번호로는 Discogs에서 못 가져왔어요",
                "hint": "숫자만 넣었는지, 아니면 .../releases/123 같은 주소인지 확인해 주세요",
            },
            ensure_ascii=False,
        )
    return json.dumps(data, ensure_ascii=False)


@tool
def wikidata_entity_specs(
    query: Annotated[str, "제품·신디·브랜드 (영문 풀네임이면 더 잘 나와요)"],
) -> str:
    """제품 이름으로 ‘백과식 사실’이 있나 Wikidata에서 찾아볼 때 써요."""
    return json.dumps(wikidata_entity_specs_fetch(query), ensure_ascii=False)


@tool
def rag_search_synth_docs(
    query: Annotated[
        str,
        "손님 상황을 담아 검색: 예산·장르·용도·키워드 (예: 저예산 하드웨어 베이스, 앰비언트 소프트)",
    ],
) -> str:
    """가게 안내집+진열 메모를 키워드로 찾아볼 때 써요. 추천할 때 감 잡기용이에요."""
    return json.dumps(_get_rag_index().search(query, k=5), ensure_ascii=False)


def _extract_api_provenance(msgs: list) -> dict[str, Any]:
    prov: dict[str, Any] = {}
    for m in msgs or []:
        if not isinstance(m, ToolMessage):
            continue
        name = getattr(m, "name", "") or ""
        raw = (m.content or "").strip()
        if not raw.startswith("{"):
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if (name == "get_discogs_release" or data.get("source") == "discogs_release_api") and data.get(
            "release_id"
        ) is not None:
            prov["discogs_release_id"] = data["release_id"]
        if name == "wikidata_entity_specs" or data.get("source") == "wikidata":
            if data.get("entity_id"):
                prov["wikidata_qid"] = data["entity_id"]
            elif data.get("note_ko"):
                prov["wikidata_note"] = data["note_ko"]
            elif data.get("error"):
                prov["wikidata_note"] = str(data["error"])
        if name == "rag_search_synth_docs" or data.get("source") == "rag":
            if data.get("hits"):
                prov["rag_chunk_ids"] = [h.get("chunk_id") for h in data["hits"] if isinstance(h, dict)]
            elif data.get("error"):
                prov["rag_note"] = str(data["error"])
    return prov


CATALOG: list[dict[str, Any]] = [
    {
        "name": "Korg NTS-1 digital kit",
        "kind": "hardware",
        "bucket": "low",
        "price_krw": 240000,
        "blurb": "작은 디지털 킷, 이펙터로도 활용. 입문 실험용.",
    },
    {
        "name": "Roland JD-08",
        "kind": "hardware",
        "bucket": "mid",
        "price_krw": 450000,
        "blurb": "JD-800 계열 바이트폴리. 패드·일렉트로닉에 어울림.",
    },
    {
        "name": "Sequential Take 5",
        "kind": "hardware",
        "bucket": "high",
        "price_krw": 1800000,
        "blurb": "아날로그 5음, 첫 폴리 아날로그로 인기.",
    },
    {
        "name": "Vital (소프트웨어)",
        "kind": "software",
        "bucket": "low",
        "price_krw": 0,
        "blurb": "웨이브테이블 무료 tier. 튜토리얼 많음.",
    },
    {
        "name": "Arturia Analog Lab V 번들",
        "kind": "software",
        "bucket": "mid",
        "price_krw": 120000,
        "blurb": "프리셋 다수, 키보드 번들에 자주 포함.",
    },
    {
        "name": "Xfer Serum",
        "kind": "software",
        "bucket": "high",
        "price_krw": 210000,
        "blurb": "웨이브테이블 표준. EDM·베이스 설계에 강함.",
    },
    {
        "name": "Yamaha reface CS",
        "kind": "hardware",
        "bucket": "mid",
        "price_krw": 520000,
        "blurb": "가벼운 VA, 배터리 구동 가능.",
    },
]


def _file_guide_rag_chunks() -> list[str]:
    path = Path(__file__).resolve().parent / "data" / "synth_rag_corpus.md"
    if path.is_file():
        raw = path.read_text(encoding="utf-8")
        parts = re.split(r"\n---\n", raw.strip())
        out = [p.strip() for p in parts if p.strip()]
        return out if out else list(_DEFAULT_RAG_CHUNKS)
    return list(_DEFAULT_RAG_CHUNKS)


def _catalog_rag_chunks() -> list[str]:
    return [
        (
            "[로컬 카탈로그·추천 RAG] "
            f"제품={it['name']}; 종류={it['kind']}; 예산대={it['bucket']}; "
            f"요약={it['blurb']}; price_krw(목 데이터)={it['price_krw']}."
        )
        for it in CATALOG
    ]


def _build_rag_chunks_split() -> tuple[list[str], int]:
    guides = _file_guide_rag_chunks()
    cat = _catalog_rag_chunks()
    return guides + cat, len(guides)


def _kind_ok(item: dict, synth_kind: str) -> bool:
    if synth_kind == "both":
        return True
    return item["kind"] == synth_kind


def _bucket_ok(item: dict, budget_band: str) -> bool:
    return item["bucket"] == budget_band


def make_tools(budget_band: str, synth_kind: str):
    @tool
    def search_catalog(
        query: Annotated[str, "찾고 싶은 말 (브랜드, 장르, 디지털/아날로그 느낌 등)"],
    ) -> str:
        """우리 가게 진열 목록 + (선택) Discogs에서 비슷한 릴리즈까지 같이 찾아봐요. 추천 전에 한번 돌려 주세요."""
        q = (query or "").lower()
        raw_q = (query or "").strip()
        hits: list[dict] = []
        for it in CATALOG:
            if not _kind_ok(it, synth_kind) or not _bucket_ok(it, budget_band):
                continue
            blob = f"{it['name']} {it['blurb']}".lower()
            if not q or q in blob or any(len(w) > 2 and w in blob for w in q.split()):
                hits.append(it)
        if not hits and q:
            for it in CATALOG:
                if _kind_ok(it, synth_kind) and _bucket_ok(it, budget_band):
                    hits.append(it)

        discogs_q = raw_q if raw_q else "electronic synthesizer"
        api_rows = _discogs_search_releases(discogs_q, per_page=8)
        api_filtered = [r for r in api_rows if synth_kind == "both" or r.get("kind") == synth_kind]

        payload: list[dict[str, Any]] = []
        for x in hits:
            payload.append(
                {
                    "name": x["name"],
                    "kind": x["kind"],
                    "price_krw": x["price_krw"],
                    "blurb": x["blurb"],
                    "bucket": x.get("bucket"),
                    "source": "local",
                }
            )
        for r in api_filtered:
            payload.append(r)

        note_parts = [
            "왼쪽 위저드에서 고른 예산·종류에 맞춰 로컬 목록만 골라 보여드렸어요.",
            "Discogs 쪽은 ‘음반/릴리즈’ 정보예요. 가격·재고랑은 별개고, 예산 필터는 안 탑니다.",
            "외부 검색 끄고 싶으면 DISCOGS_SEARCH=0 으로 꺼 주시면 돼요.",
        ]
        if not payload:
            return json.dumps(
                {
                    "matches": [],
                    "note": "지금 조건으로는 딱 맞는 게 안 보여요. 질문을 조금 넓혀 보시거나 위저드 설정을 한번 봐 주세요. "
                    "Discogs도 텅 비면 영문 브랜드명으로 다시 검색해 볼까요?",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"matches": payload, "note": " ".join(note_parts)},
            ensure_ascii=False,
        )

    @tool
    def compare_synths(names: Annotated[str, "진열 이름 그대로, 쉼표로 몇 개"]) -> str:
        """진열 목록에 있는 제품 이름이 글자 그대로 같을 때만 나란히 비교해 드려요. Discogs만 나온 건 search_catalog JSON 보고 말로 풀어 드리면 돼요."""
        want = [n.strip() for n in names.split(",") if n.strip()]
        rows = [it for it in CATALOG if it["name"] in want]
        if len(rows) < 2:
            return json.dumps(
                {
                    "error": "진열 목록에서 이름이 똑같이 맞는 게 두 개 이상 있어야 비교가 돼요. "
                    "Discogs에서만 본 이름이면, 제가 글로 장단점을 풀어서 말씀드릴게요.",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            [{k: r[k] for k in ("name", "kind", "bucket", "price_krw", "blurb")} for r in rows],
            ensure_ascii=False,
        )

    @tool
    def web_hint_synth(topic: Annotated[str, "손님이 더 알고 싶어 하는 주제 한 줄"]) -> str:
        """진짜 웹은 안 타고, 매장에서 흔히 하는 말로만 힌트 드려요 (MVP)."""
        t = topic.lower()
        if "가격" in t or "시세" in t:
            return "[매장 팁] 중고·번들가는 시즌마다 들쭉날쭉해요. 여기 적힌 price_krw는 연습용 숫자라 참고만 해 주세요."
        if "펌웨어" in t or "업데이트" in t:
            return "[매장 팁] 펌웨어랑 매뉴얼은 제조사 지원 페이지가 제일 정확해요. 한번 들러 보세요."
        return f"[매장 팁] '{topic}'는 공식 사이트랑 유튜브 데모를 같이 보면 감이 빨리 와요."

    return [
        search_catalog,
        get_discogs_release,
        wikidata_entity_specs,
        rag_search_synth_docs,
        compare_synths,
        explain_beginner,
        web_hint_synth,
    ]


def system_prompt_for_route(route: str, budget_band: str, synth_kind: str) -> str:
    tone = {
        "low": "손님이 부담 없이 시작할 수 있게, 가성비랑 ‘배우는 데 아깝지 않은지’를 먼저 봐 주세요.",
        "mid": "손님 예산 한가운데에서, 앞으로 쓸 일이 생각보다 많을지·장르에 맞는지 균형 있게 짚어 주세요.",
        "high": "여유 있게 고르시는 분이니, 오래 붙들고 쓸 만한지·나중에 확장할 여지까지 같이 이야기해 주세요.",
    }.get(route, "예산 한가운데 기준으로 편하게 상담해 주세요.")
    return f"""당신은 동네 악기 가게 직원처럼, 한국어 **해요체**로 손님께 신디를 추천·설명합니다.
말투는 딱딱한 보고서가 아니라 매장에서 말 건네듯 친근하게. 전문 용어는 짧게 풀어서 곁들이세요.
{tone}
위저드에서 정한 값: 예산구간={budget_band}, 신디종류={synth_kind}.
일하는 순서 (seed v2 + RAG):
1) 추천·비교면 먼저 rag_search_synth_docs에 예산·장르·하드/소프트 의도를 담아 검색하세요. hits가 로컬카탈로그면 그 제품명은 search_catalog로 한 번 더 맞춰 보세요.
2) 그다음 search_catalog로 우리 진열+Discogs 후보를 받으세요. discogs_release_id가 있으면 필요할 때만 get_discogs_release로 더 파보세요.
3) 제품·브랜드 사실은 wikidata_entity_specs로 확인하세요. JSON에 없는 숫자·스펙은 억지로 채우지 마세요.
4) get_discogs_release는 음반 릴리즈 정보예요. 매장 가격이나 하드웨어 스펙 시트와 다를 수 있어요.
5) Wikidata가 비면 “제가 찾은 백과에는 없어요”라고 솔직히 말하세요.
6) 진열 목록에 없는 가격은 지어내지 마세요. compare_synths는 로컬 이름이 글자 그대로 같을 때만 쓰세요.
7) 용어는 explain_beginner, 더 필요하면 rag_search_synth_docs·web_hint_synth를 섞어 쓰세요.
8) 최종 답은 아래 **마크다운 제목 순서**를 꼭 지키세요 (내용은 친근한 상담 톤):
## 한마디로
## 이런 건 어떠세요
## 제가 참고한 건…
## 처음이시면
## 솔직히 말씀드리면"""


def build_react_agent_for_session(route: str, budget_band: str, synth_kind: str):
    llm = ChatOpenAI(model="gpt-5-mini")
    tools = make_tools(budget_band, synth_kind)
    return create_react_agent(
        llm,
        tools,
        prompt=system_prompt_for_route(route, budget_band, synth_kind),
    )


class SynthGraphState(TypedDict, total=False):
    messages: list
    budget_band: str
    synth_kind: str
    use_hints: list[str]
    route: str
    last_recommendation: dict[str, Any]
    last_api_provenance: dict[str, Any]


def router_node(state: SynthGraphState) -> dict:
    bb = state.get("budget_band") or "mid"
    if bb not in ("low", "mid", "high"):
        bb = "mid"
    return {"route": bb}


def agent_node(state: SynthGraphState) -> dict:
    route = state.get("route") or "mid"
    band = state.get("budget_band") or "mid"
    kind = state.get("synth_kind") or "both"
    hints = state.get("use_hints") or []
    agent = build_react_agent_for_session(route, band, kind)
    ctx = HumanMessage(
        content=(
            f"[매장 상담 맥락] 지금 손님 설정이에요 — 분기={route}, 예산={band}, 종류={kind}, 힌트={hints}. "
            "아래 말씀에 악기상처럼 편하게 답해 주세요."
        )
    )
    before = list(state.get("messages") or [])
    inp = [ctx] + before
    res = agent.invoke({"messages": inp})
    tail = res["messages"][len(inp) :]
    merged = before + tail
    last_ai = ""
    for m in reversed(merged):
        if isinstance(m, AIMessage) and (m.content or "").strip():
            last_ai = m.content.strip()
            break
    last_rec: dict[str, Any] = {
        "summary_text": last_ai[:2000] if last_ai else "",
        "route": route,
        "budget_band": band,
        "synth_kind": kind,
    }
    prov = _extract_api_provenance(merged)
    return {
        "messages": merged,
        "last_recommendation": last_rec,
        "last_api_provenance": prov,
    }


def route_edge(state: SynthGraphState) -> Literal["low", "mid", "high"]:
    r = state.get("route") or "mid"
    if r not in ("low", "mid", "high"):
        return "mid"
    return r  # type: ignore[return-value]


def compile_synth_workflow():
    g = StateGraph(SynthGraphState)
    g.add_node("router", router_node)
    g.add_node("agent", agent_node)
    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        route_edge,
        {"low": "agent", "mid": "agent", "high": "agent"},
    )
    g.add_edge("agent", END)
    return g.compile()


def run_graph(
    messages: list,
    budget_band: str,
    synth_kind: str,
    use_hints: list[str],
    workflow,
):
    return workflow.invoke(
        {
            "messages": messages,
            "budget_band": budget_band,
            "synth_kind": synth_kind,
            "use_hints": use_hints,
            "route": "",
        }
    )


def _last_ai_text(messages: list) -> str:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and (m.content or "").strip():
            return m.content.strip()
    return "잠깐, 답이 비어 있네요. 한 번만 다시 말씀해 주시겠어요?"


def run_smoke() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 없어요 (.env 확인)", file=sys.stderr)
        sys.exit(1)
    wf = compile_synth_workflow()
    msgs = [HumanMessage(content="저예산 하드웨어 하나만 가게 목록에서 찾아서 추천해 주세요.")]
    out = run_graph(msgs, "low", "hardware", [], wf)
    print("route=", out.get("route"))
    print("last_recommendation keys=", list((out.get("last_recommendation") or {}).keys()))
    print("reply preview=", _last_ai_text(out.get("messages") or [])[:900])


def run_smoke_api() -> None:
    """OpenAI 없이 Discogs·Wikidata HTTP만 검증."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    rows = _discogs_search_releases("korg", 1)
    print("discogs search rows:", len(rows))
    if rows and rows[0].get("discogs_release_id"):
        rid = rows[0]["discogs_release_id"]
        rel = discogs_fetch_release(str(rid))
        print("discogs release title:", (rel or {}).get("title"), "tracks:", (rel or {}).get("track_count"))
    wd = wikidata_entity_specs_fetch("Korg MS-20")
    print("wikidata qid:", wd.get("entity_id"), "claims keys:", list((wd.get("claims") or {}).keys())[:8])


def _streamlit_main() -> None:
    st.set_page_config(page_title="신디, 뭐 살지 같이 볼까요?", page_icon="🎹")
    st.title("🎹 신디 고르기, 제가 옆에서 도와드릴게요")
    st.caption("가게 메모 + Discogs·Wikidata까지 (seed.yaml v2)")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("열쇠가 없네요. `.env`에 OPENAI_API_KEY를 넣어 주세요.")
        st.stop()

    if "lc_messages" not in st.session_state:
        st.session_state.lc_messages = []
    if "ui_chat" not in st.session_state:
        st.session_state.ui_chat = []
    if "last_route" not in st.session_state:
        st.session_state.last_route = ""
    if "last_recommendation" not in st.session_state:
        st.session_state.last_recommendation = {}
    if "last_api_provenance" not in st.session_state:
        st.session_state.last_api_provenance = {}
    if "workflow" not in st.session_state:
        st.session_state.workflow = compile_synth_workflow()

    with st.sidebar:
        st.header("먼저 여쭤볼게요")
        band_kr = st.selectbox(
            "대략 어느 쪽 예산이세요? (처음 분기)",
            ["저가 (~30만 원대)", "중가 (30~80만 원대)", "고가 (80만 원 이상)"],
        )
        band = {"저가 (~30만 원대)": "low", "중가 (30~80만 원대)": "mid", "고가 (80만 원 이상)": "high"}[
            band_kr
        ]
        sk = st.radio("하드웨어 / 소프트웨어", ["하드웨어", "소프트웨어", "둘 다"], horizontal=True)
        synth_kind = {"하드웨어": "hardware", "소프트웨어": "software", "둘 다": "both"}[sk]
        genre = st.text_input("장르나 쓰실 용도 (있으면)", placeholder="예: 앰비언트, 베이스 연습")
        hints = [genre.strip()] if genre.strip() else []
        if st.button("대화 다시 시작"):
            st.session_state.lc_messages = []
            st.session_state.ui_chat = []
            st.session_state.last_route = ""
            st.session_state.last_recommendation = {}
            st.session_state.last_api_provenance = {}
            st.rerun()
        st.divider()
        st.markdown(
            "**안에서 도는 흐름:** `START` → `router` → `low|mid|high` → `agent`(ReAct) → `END`"
        )
        with st.expander("Discogs / Wikidata / RAG가 뭐예요?"):
            st.markdown(
                "- [Discogs API](https://www.discogs.com/developers): 음반·릴리즈 쪽 정보예요. **시세랑은 별개**예요. 상세는 `get_discogs_release`.\n"
                "- `.env`: `DISCOGS_PERSONAL_TOKEN`은 있으면 좋고요 · 끄려면 `DISCOGS_SEARCH=0`\n"
                "- [Wikidata](https://www.wikidata.org/wiki/Wikidata:Data_access): 백과에 적힌 사실만 `wikidata_entity_specs`로.\n"
                "- **RAG**: 안내집 MD + **진열 메모 요약**을 같이 임베딩 · `rag_search_synth_docs` · `text-embedding-3-small`"
            )

    for role, text in st.session_state.ui_chat:
        with st.chat_message(role):
            st.markdown(text)

    user_text = st.chat_input("편하게 물어보세요 (예: 소프트로 바꿔서 추천해 주세요)")
    if not user_text:
        return

    st.session_state.ui_chat.append(("user", user_text))
    st.session_state.lc_messages.append(HumanMessage(content=user_text))

    with st.chat_message("assistant"):
        with st.spinner("잠깐만요, 찾아보고 올게요…"):
            try:
                out = run_graph(
                    st.session_state.lc_messages,
                    band,
                    synth_kind,
                    hints,
                    st.session_state.workflow,
                )
                st.session_state.last_route = out.get("route", "")
                st.session_state.lc_messages = list(out.get("messages") or [])
                st.session_state.last_recommendation = dict(out.get("last_recommendation") or {})
                st.session_state.last_api_provenance = dict(out.get("last_api_provenance") or {})
                reply = _last_ai_text(st.session_state.lc_messages)
            except Exception as e:
                reply = f"앗, 여기서 막혔어요: {e}"
        st.markdown(reply)
        with st.expander("지금 어떻게 돌아갔는지 (라우팅·출처)"):
            st.code(
                json.dumps(
                    {
                        "budget_band": band,
                        "synth_kind": synth_kind,
                        "use_hints": hints,
                        "route_after_router": st.session_state.last_route,
                        "last_recommendation": st.session_state.last_recommendation,
                        "last_api_provenance": st.session_state.last_api_provenance,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                language="json",
            )

    st.session_state.ui_chat.append(("assistant", reply))


if __name__ == "__main__":
    if "--smoke-api" in sys.argv:
        run_smoke_api()
    elif "--smoke" in sys.argv:
        run_smoke()
    else:
        _streamlit_main()
