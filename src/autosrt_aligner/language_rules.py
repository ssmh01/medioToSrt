"""Language-specific word and phrase boundaries for subtitle segmentation."""

from __future__ import annotations
import threading
from functools import lru_cache
import re
from pathlib import Path

from .errors import AlignmentError

DATA_DIR = Path(__file__).resolve().parent / "data"
ANALYSIS_LOCK = threading.Lock()
CLOSERS = set("\\\"'」』”’）)]】")
STRONG = set("。！？!?")
MID = set("，、,;；:：")
SPOKEN = re.compile(r"[A-Za-z0-9\u3400-\u9fff\u3040-\u30ff\uac00-\ud7a3]")
ABBR = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "a.m",
    "p.m",
    "u.s",
    "u.k",
    "e.g",
    "i.e",
    "ph.d",
}


def opening_positions(text):
    pairs = {
        "「": "」",
        "『": "』",
        "“": "”",
        "‘": "’",
        "（": "）",
        "(": ")",
        "[": "]",
        "【": "】",
    }
    stack = []
    opens = set()
    for i, c in enumerate(text):
        if c in pairs:
            opens.add(i)
            stack.append(pairs[c])
        elif c in {'"', "'"}:
            if (
                c == "'"
                and i
                and i + 1 < len(text)
                and text[i - 1].isalnum()
                and text[i + 1].isalnum()
            ):
                continue
            if stack and stack[-1] == c:
                stack.pop()
            else:
                opens.add(i)
                stack.append(c)
        elif stack and c == stack[-1]:
            stack.pop()
    return opens


def display_boundaries(text, tokens):
    opens = opening_positions(text)
    result = []
    for i, t in enumerate(tokens):
        if i + 1 == len(tokens):
            result.append(len(text))
            continue
        end = tokens[i + 1].start_char
        attached = [p for p in opens if (t.end_char or 0) <= p < end]
        result.append(min(attached) if attached else end)
    return result


def sentence_ends(text, language):
    ends = []
    opens = opening_positions(text)
    for m in re.finditer(r"[。！？!?]+|(?<!\.)\.(?!\.)", text):
        a, b = m.span()
        if text[a] == ".":
            if a and b < len(text) and text[a - 1].isdigit() and text[b].isdigit():
                continue
            w = re.search(r"([A-Za-z.]+)$", text[:a])
            if w and (
                w.group(1).lower() in ABBR
                or (len(w.group(1)) == 1 and w.group(1).isupper())
            ):
                continue
            if language in {"en", "ko"} and b < len(text) and text[b].isalnum():
                continue
        while (
            b < len(text)
            and b not in opens
            and (text[b].isspace() or text[b] in CLOSERS)
        ):
            b += 1
        if SPOKEN.search(text[b:]):
            ends.append(b)
    return sorted(set(ends))


def boundary_kind(text, end, language):
    left = text[:end].rstrip()
    while left and left[-1] in CLOSERS:
        left = left[:-1].rstrip()
    if not left:
        return "weak"
    if left[-1] in STRONG:
        return "sentence"
    if left[-1] == ".":
        w = re.search(r"([A-Za-z.]+)\.$", left)
        if not w or w.group(1).lower() not in ABBR:
            return "sentence"
    if left[-1] in MID:
        return "clause"
    return "weak"


@lru_cache(maxsize=3)
def _model(lang):
    if lang == "zh":
        import jieba
        import jieba.posseg

        return jieba.posseg.POSTokenizer(
            jieba.Tokenizer(str(DATA_DIR / "dict.txt.big"))
        )
    if lang == "ja":
        from sudachipy import dictionary

        return dictionary.Dictionary().create()
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner"])


def analyze(text, lang):
    # Sudachi's native tokenizer is shared by concurrent web jobs.
    with ANALYSIS_LOCK:
        try:
            return _analyze(text, lang)
        except (ImportError, OSError) as exc:
            raise AlignmentError(
                "字幕分词依赖不可用，请安装 requirements.txt 中的依赖: " + str(exc)
            ) from exc


def _analyze(text, lang):
    words = []
    atoms = []
    if lang == "ko":
        return {
            "words": words,
            "atoms": atoms,
            "sentence_ends": sentence_ends(text, lang),
        }
    model = _model(lang)

    def atom(a, b, kind, allow_comma=False):
        if b > a and not any(
            c in STRONG or (c in MID and not (allow_comma and c in ",，、"))
            for c in text[a:b]
        ):
            atoms.append({"start": a, "end": b, "kind": kind})

    if lang == "zh":
        cursor = 0
        for token in model.cut(text, HMM=True):
            a = cursor
            b = a + len(token.word)
            if text[a:b] != token.word:
                raise ValueError("Chinese offset mismatch")
            cursor = b
            if SPOKEN.search(token.word):
                words.append(
                    {"start": a, "end": b, "text": token.word, "pos": token.flag}
                )
                atom(a, b, "word")
        for l, r in zip(words, words[1:]):
            if text[l["end"] : r["start"]].strip():
                continue
            if (l["pos"] == "m" and r["pos"].startswith("q")) or r["text"] in {
                "的",
                "地",
                "得",
                "了",
                "着",
                "过",
            }:
                atom(l["start"], r["end"], "number_or_suffix")
            if l["text"] == "的" and r["pos"].startswith("n"):
                atom(l["start"], r["end"], "modifier_noun")
            if l["text"] in {"不", "没", "未", "别"} and r["pos"].startswith(
                ("v", "a")
            ):
                atom(l["start"], r["end"], "negation")
    elif lang == "ja":
        from sudachipy import tokenizer

        # Sudachi has a per-call byte limit; sentence-sized chunks retain offsets.
        for match in re.finditer(r"[^。！？!?\n]+[。！？!?\n]*|[。！？!?\n]+", text):
            offset = match.start()
            for m in model.tokenize(match.group(), tokenizer.Tokenizer.SplitMode.C):
                a = offset + m.begin()
                b = offset + m.end()
                pos = list(m.part_of_speech())
                if SPOKEN.search(text[a:b]):
                    words.append({"start": a, "end": b, "text": text[a:b], "pos": pos})
                    atom(a, b, "word")
        for l, r in zip(words, words[1:]):
            if text[l["end"] : r["start"]].strip(' \t\r\n」』”’"'):
                continue
            if r["pos"][0] in {"助詞", "助動詞", "接尾辞"}:
                atom(l["start"], r["end"], "attached_morpheme")
            if (
                r["pos"][0] == "動詞"
                and r["pos"][1] == "非自立可能"
                and l["pos"][0] in {"動詞", "助動詞", "助詞"}
            ):
                atom(l["start"], r["end"], "verb_chain")
            if (
                l["pos"][0] in {"形容詞", "連体詞", "接頭辞"}
                or l["pos"][5].startswith("連体形")
            ) and r["pos"][0] == "名詞":
                atom(l["start"], r["end"], "modifier_noun")
    else:
        doc = model(text)
        for t in doc:
            if SPOKEN.search(t.text):
                words.append(
                    {
                        "start": t.idx,
                        "end": t.idx + len(t),
                        "text": t.text,
                        "pos": t.pos_,
                        "dep": t.dep_,
                        "head": t.head.idx,
                    }
                )
                atom(t.idx, t.idx + len(t), "word")
            if t.dep_ in {
                "det",
                "amod",
                "compound",
                "nummod",
                "poss",
                "aux",
                "auxpass",
                "neg",
                "prt",
                "case",
                "fixed",
                "flat",
            } or (t.dep_ == "nsubj" and t.pos_ == "PRON" and abs(t.i - t.head.i) <= 2):
                a = min(t.idx, t.head.idx)
                b = max(t.idx + len(t), t.head.idx + len(t.head))
                if b - a <= 80:
                    atom(a, b, "dependency_" + t.dep_)
            if t.dep_ == "prep" and t.head.pos_ in {"NOUN", "PROPN"}:
                objects = [x for x in t.children if x.dep_ == "pobj"]
                if objects:
                    a = t.head.idx
                    b = max(x.idx + len(x) for x in objects)
                    if 0 < b - a <= 45:
                        atom(a, b, "noun_prepositional_modifier")
            if (
                t.dep_ == "conj"
                and t.pos_ in {"NOUN", "PROPN"}
                and t.head.pos_ in {"NOUN", "PROPN"}
            ):
                a = min(t.idx, t.head.idx)
                b = max(t.idx + len(t), t.head.idx + len(t.head))
                if (
                    b - a <= 60
                    and "," in text[a:b]
                    and not any(
                        x.pos_ in {"VERB", "AUX"} and a <= x.idx < b for x in doc
                    )
                ):
                    atom(a, b, "coordinated_noun", allow_comma=True)
    return {"words": words, "atoms": atoms, "sentence_ends": sentence_ends(text, lang)}


def make_boundary_map(text, lang, analysis):
    n = len(text)
    word_inside = [False] * (n + 1)
    phrase_inside = [False] * (n + 1)
    for a in analysis["atoms"]:
        dest = word_inside if a["kind"] in {"word", "eojeol"} else phrase_inside
        for p in range(a["start"] + 1, a["end"]):
            dest[p] = True
    ws = analysis["words"]
    next_word = [None] * (n + 1)
    j = len(ws) - 1
    following = None
    for p in range(n, -1, -1):
        while j >= 0 and ws[j]["start"] >= p:
            following = ws[j]
            j -= 1
        next_word[p] = following
    return {
        "word_inside": word_inside,
        "phrase_inside": phrase_inside,
        "following": next_word,
    }


def lexical_penalty(lang, p, maps):
    if maps["word_inside"][p]:
        return 160.0
    if maps["phrase_inside"][p]:
        return 85.0
    r = maps["following"][p]
    if lang == "en" and r:
        if r["pos"] in {"CCONJ", "SCONJ", "ADP"}:
            return 7.0
        return 23.0
    if lang == "zh" and r:
        if r["pos"] in {"c", "p"}:
            return 12.0
        return 30.0
    return 30.0
