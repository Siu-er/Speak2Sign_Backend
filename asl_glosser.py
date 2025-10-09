# asl_glosser.py

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from wordfreq import zipf_frequency
import unicodedata
import contractions
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@dataclass
class GlossResult:
    gloss_tokens: List[str]
    gloss: str
    sentence_nmm: Dict[str, Optional[str]] = field(default_factory=dict)


class ASLGlosser:
    def __init__(self, data_dir: str):
        """
        Initialize ASL Glosser with data files

        Args:
            data_dir: Directory containing lexicon.json, config.json
        """
        self.data_dir = data_dir
        self.T5_MODEL_NAME = "google/flan-t5-small"
        self.T5_MAX_NEW_TOKENS = 64

        # Load configuration files
        with open(os.path.join(data_dir, "lexicon.json"), "r", encoding="utf-8") as f:
            self.LEXICON: Dict[str, Optional[str]] = json.load(f)
        with open(os.path.join(data_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(self.data_dir, "patterns.json"), "r", encoding="utf-8") as f:
            self._pattern_cfg = json.load(f)

        # Config (keep only ASL policy knobs; morphology handled by spaCy)
        self.FUNCTION_WORDS = set(cfg["function_words"])       # curated set to drop
        self.WH = set(cfg["wh_words"])                         # {"who","what","where","when","why","how"}
        self.FUNCTION_WORDS -= self.WH
        self.UNIT_MAP = cfg["units_map"]                       # lemma -> GLOSS, e.g., {"minute":"MINUTE"}
        self.FRONT_TIME = set(cfg["front_time"])               # tokens to front, e.g., {"TOMORROW", "TODAY"}

        # Inline small language knobs (no phrases.json needed)
        self.AM_PM = set(cfg["am_pm"])
        self.PRE_INTENS = set(cfg["pre_intensifiers"])
        self.PRONOUNS = set(cfg["protected_pronouns"])
        self.POSS_PRONOUN_MAP = dict(cfg["possessive_pronoun_map"])
        self.WH_SUBORDINATORS = set(cfg["wh_subordinators"])
        self.MODALS_REORDER = set(cfg["modals_reorder"])
        self.AUX_SURFACE_DROP = set(cfg["aux_surface_drop"])

        # policy: never drop pronouns/WH even if listed by mistake
        self.FUNCTION_WORDS -= self.PRONOUNS
        self.FUNCTION_WORDS -= self.WH

        # spaCy pipeline (keep parser for POS/DEP; disable heavy components)
        self._nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

        # spaCy Matcher for MWEs & post-intensifiers
        self._matcher = Matcher(self._nlp.vocab)

        self._pattern_actions = {}
        for item in self._pattern_cfg.get("token_patterns", []):
            name = item["name"]
            pattern = item["pattern"]
            action = item["action"]
            # spaCy Matcher expects a list of patterns (OR). We wrap single pattern.
            self._matcher.add(name, [pattern])
            self._pattern_actions[name] = action

        # --- T5 corrector (optional) ---
        self.t5_enabled = False
        self.t5_tokenizer = None
        self.t5_model = None
        self.t5_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5_max_new_tokens = self.T5_MAX_NEW_TOKENS

        if self.T5_MODEL_NAME:
            # good defaults: "google/flan-t5-small" (instruction-tuned) or your fine-tuned checkpoint
            self.t5_tokenizer = AutoTokenizer.from_pretrained(self.T5_MODEL_NAME)
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(self.T5_MODEL_NAME).to(self.t5_device)
            # self.t5_enabled = True

    # ---------------------------
    # Helpers
    # ---------------------------

    # Turn on with: glosser.DEBUG = True
    DEBUG = True

    def _dbg(self, *args):
        if self.DEBUG:
            print("[ASL DEBUG]", *args)

    def _dump_doc(self, doc, note=""):
        if not self.DEBUG:
            return
        print("\n[ASL DEBUG] ---- DOC DUMP", note, "----")
        print("text:", repr(doc.text))
        for i, t in enumerate(doc):
            print(
                f"{i:02d}  txt={t.text!r:14} lem={t.lemma_!r:10} pos={t.pos_:5} tag={t.tag_:5} dep={t.dep_:6} head={t.head.text!r:10} morph={t.morph}")
        print("[ASL DEBUG] --------------\n")

    def _ml_refine_gloss(self, src_text: str, draft_gloss: str, qtype: Optional[str]) -> str:
        """
        Use T5 to post-process the rule-based gloss.
        Keeps behavior deterministic by sanitizing tokens after generation.
        """
        if not self.t5_enabled:
            return draft_gloss

        # Compact, explicit prompt; include QTYPE so T5 doesn’t drop the question mark or WH
        prompt = (
            "Rewrite the ASL gloss to be grammatical, preserve meaning and tokens when possible.\n"
            f"EN: {src_text}\n"
            f"QTYPE: {qtype or 'NONE'}\n"
            f"DRAFT: {draft_gloss}\n"
            "FINAL:"
        )

        enc = self.t5_tokenizer(prompt, return_tensors="pt", truncation=True).to(self.t5_device)
        with torch.no_grad():
            out_ids = self.t5_model.generate(
                **enc,
                max_new_tokens=self.t5_max_new_tokens,
                num_beams=4,
                length_penalty=0.1,
                early_stopping=True
            )
        txt = self.t5_tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Sanitize: uppercase tokens, collapse whitespace, strip stray punctuation
        gloss = re.sub(r"\s+", " ", txt.strip()).upper()

        # Ensure single trailing question mark for questions
        if qtype in {"WH", "YN"}:
            gloss = gloss.rstrip()
            if not gloss.endswith("?"):
                # prefer “ ... WHERE ? ” style; if a ? already inside, leave it
                if "?" not in gloss:
                    gloss = gloss + " ?"

        # Disallow commas/periods in gloss, keep digits/AM/PM/dashes and '?'
        gloss = re.sub(r"[^\w\-\s\?]", " ", gloss)
        gloss = re.sub(r"\s+", " ", gloss).strip()
        return gloss

    def _is_relative_wh(self, tok) -> bool:
        """Return True if 'who/what/where/...' is used in a relative clause (not a question)."""
        if tok.lower_ not in self.WH:
            return False
        # Don't treat as relative if the whole sentence is a question
        if tok.doc.text.strip().endswith("?"):
            return False
        # If any ancestor is a relative clause, it's a relative WH (drop it)
        return any(anc.dep_ == "relcl" for anc in tok.ancestors)

    def reorder_modal_inversion(self, tokens, qtype):
        if qtype != "YN":
            return tokens
        pronouns = {"ME", "YOU", "WE", "HE", "SHE", "THEY", "IT"}
        modals = self.MODALS_REORDER
        try:
            m_idx = next(i for i, t in enumerate(tokens) if t in modals)
            p_idx = next(i for i, t in enumerate(tokens) if t in pronouns)
        except StopIteration:
            return tokens
        if m_idx < p_idx:
            modal = tokens.pop(m_idx)
            if m_idx < p_idx:
                p_idx -= 1
            tokens.insert(p_idx + 1, modal)
        return tokens

    def expand_contractions(self, text: str) -> str:
        """Expand English contractions and normalize quotes."""
        t = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
        return contractions.fix(t, slang=True)

    def _is_likely_english_word(self, word: str) -> bool:
        """Zipf > 3.0 ≈ likely real English word."""
        if not word or not re.fullmatch(r"[a-z]+(?:-[a-z]+)?", word):
            return False
        return zipf_frequency(word, "en", minimum=0.0) >= 3.0

    def detect_qtype(self, doc, raw_text: str) -> Optional[str]:
        s = raw_text.strip()

        # If it ends with a '?', classify as WH vs YN by presence of WH
        if s.endswith("?"):
            for tok in doc:
                t = tok.lower_
                if t in self.WH:
                    # If it's a clause-marker *subordinator*, don't count as interrogative WH
                    if tok.dep_ == "mark" and t in self.WH_SUBORDINATORS:
                        continue
                    return "WH"
            return "YN"

        # No '?': only YN if AUX-initial (e.g., 'Do you...', 'Are you...')
        if len(doc) and doc[0].pos_ == "AUX":
            return "YN"

        return None

    def detect_negation_doc(self, doc) -> bool:
        """Use dependency 'neg' or lexical negators."""
        return any(t.dep_ == "neg" or t.lower_ in {"not", "never", "no"} for t in doc)

    def move_wh_to_end(self, tokens: List[str]) -> List[str]:
        """Move WH-words to sentence-final position (ASL WH questions)."""
        wh = [t for t in tokens if t in {"WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW"}]
        rest = [t for t in tokens if t not in {"WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW"}]
        return rest + wh

    def place_negation(self, tokens: List[str]) -> List[str]:
        """Insert NOT after pronoun if present, else prefix."""
        pron = {"ME", "YOU", "WE", "HE", "SHE", "THEY", "IT"}
        for i, t in enumerate(tokens):
            if t in pron:
                return tokens[:i+1] + ["NOT"] + tokens[i+1:]
        return ["NOT"] + tokens

    def front_time_topic(self, gloss_tokens: List[str]) -> List[str]:
        """Front time/topic expressions."""
        front = [g for g in gloss_tokens if g in self.FRONT_TIME]
        rest = [g for g in gloss_tokens if g not in self.FRONT_TIME]
        return front + rest

    def collapse_duplicates(self, toks: List[str]) -> List[str]:
        out = []
        for t in toks:
            if out and out[-1] == t:
                continue
            out.append(t)
        return out

    def map_token_spacy(self, tok) -> Optional[str]:
        lemma = tok.lemma_.lower()
        text = tok.text
        doc_ends_q = tok.doc.text.strip().endswith("?")

        # --- WH handling: keep only in real questions ---
        if tok.lower_ in self.WH:
            if not doc_ends_q:
                # Declaratives: drop WH (covers subordinators, relatives, embedded content WH)
                return None
            # Real question: keep WH (moved to end later)
            return tok.lower_.upper()

        # Skip lexical negators; structural NOT handles negation
        if tok.dep_ == "neg" or tok.lower_ in {"not", "never", "no"}:
            return None

        # Drop surface auxiliaries we don't gloss (keep true modals elsewhere)
        if tok.pos_ == "AUX" and tok.lemma_.lower() in self.AUX_SURFACE_DROP:
            return None

        # Drop function words first (policy wins)
        if lemma in self.FUNCTION_WORDS:
            return None

        # Lexicon hit on lemma
        if lemma in self.LEXICON:
            val = self.LEXICON[lemma]
            return None if val is None else val

        # Numbers / existing ASL-style
        if tok.like_num:
            return text
        if "-" in text and text.upper() == text:
            return text

        # Proper nouns -> fingerspell
        if tok.pos_ == "PROPN":
            up = re.sub(r"[^A-Za-z]", "", text).upper()
            return f"FS-{up}" if up else None

        # English-looking word -> uppercase gloss
        if lemma.isalpha() and self._is_likely_english_word(lemma):
            return lemma.upper()

        # Fallback fingerspelling
        if text.isalpha():
            return f"FS-{text.upper()}"
        return None

    # ---------------------------
    # Main pipeline
    # ---------------------------

    def gloss(self, text: str) -> GlossResult:
        """
        Convert English text to ASL gloss

        Steps:
        - Expand contractions
        - spaCy parse for lemmas/pos/dep
        - Detect question type + negation
        - Apply Matcher spans (FUTURE/MUST/etc.)
        - Map tokens with rules (intensifiers, numbers+units, perfect 'FINISH')
        - Possessives: emit before noun by default; defer to after head noun
          when the previous non-fronted token is the same pronoun (to avoid ME ME)
        - Apply ASL structure (time-fronting, YN inversion fix, neg placement, WH movement)
        """
        # 1) Normalize & expand contractions
        expanded = self.expand_contractions(text)

        # 2) Parse with spaCy
        doc = self._nlp(expanded)

        # Optional debug
        self._dump_doc(doc, note="after expand_contractions")
        self._dbg("FUNCTION_WORDS has 'my'?", "my" in self.FUNCTION_WORDS)
        self._dbg("POSS_PRONOUN_MAP:", getattr(self, "POSS_PRONOUN_MAP", {}))

        # 3) Question type + negation (reuse same doc; no double-parse)
        qtype = self.detect_qtype(doc, expanded)
        neg = self.detect_negation_doc(doc)

        # 4) Run matcher; keep longest non-overlapping spans
        matches = self._matcher(doc)
        spans = filter_spans([doc[s:e] for (_, s, e) in matches])

        # start -> (end, gloss, emit_head_verb)
        span_action_by_start: Dict[int, tuple] = {}
        post_intens_by_start: Dict[int, int] = {}

        for mid, s, e in matches:
            name = self._nlp.vocab.strings[mid]
            if not any(sp.start == s and sp.end == e for sp in spans):
                continue
            action = self._pattern_actions.get(name, {})
            atype = action.get("type")
            if atype == "emit_gloss":
                gloss = action.get("gloss")
                emit_head = bool(action.get("emit_head_verb", False))
                if gloss:
                    span_action_by_start[s] = (e, gloss, emit_head)
            elif atype == "post_intens":
                # amount currently unused (we just ++ once per match)
                post_intens_by_start[s] = e

        # 5) Token walk (+ adaptive possessive handling)
        mapped: List[str] = []
        plus_next = False

        # Possessives we defer (keyed by head token index)
        pending_poss_after: Dict[int, List[str]] = {}

        def last_non_fronted(tok_list: List[str]) -> Optional[str]:
            for t in reversed(tok_list):
                if t not in self.FRONT_TIME:
                    return t
            return None

        i = 0
        N = len(doc)
        while i < N:
            tok = doc[i]

            # Skip pure punctuation/space
            if tok.is_punct or tok.is_space:
                i += 1
                continue

            # --- Possessive determiners ---
            # Fire if spaCy flags possessive determiner (dep 'poss') OR PRP$ OR morph Poss=Yes
            if (tok.dep_ == "poss") or (tok.tag_ == "PRP$") or ("Yes" in tok.morph.get("Poss")):
                key = tok.lower_
                poss = self.POSS_PRONOUN_MAP.get(key) or self.POSS_PRONOUN_MAP.get(tok.lemma_.lower())
                if poss:
                    prev_nf = last_non_fronted(mapped)
                    # If previous non-fronted token is the same pronoun (e.g., earlier 'ME'),
                    # defer possessive to after its head to avoid ME ME adjacency after fronting.
                    if prev_nf == poss:
                        self._dbg(f"schedule poss (defer): txt={tok.text!r} -> {poss} after head idx {tok.head.i}")
                        pending_poss_after.setdefault(tok.head.i, []).append(poss)
                    else:
                        self._dbg(f"emit poss (immediate): txt={tok.text!r} -> {poss}")
                        mapped.append(poss)
                i += 1
                continue

            # If a gloss span starts here -> emit gloss and skip span
            if i in span_action_by_start:
                end, gloss, emit_head = span_action_by_start[i]

                # apply any pending pre-intensifier to the gloss itself
                if plus_next:
                    gloss = gloss + "++"
                    plus_next = False
                mapped.append(gloss)

                # also emit the head verb (last token of the span), e.g., 'study' in "need to study"
                if emit_head:
                    head_tok = doc[end - 1]
                    g2 = self.map_token_spacy(head_tok)
                    if g2:
                        mapped.append(g2)

                i = end
                continue

            # If a post-intensifier span starts here -> boost previous and skip
            if i in post_intens_by_start:
                if mapped:
                    mapped[-1] = mapped[-1] + "++"
                i = post_intens_by_start[i]
                continue

            # Pre-intensifier word: flag next mapped token
            if tok.lemma_.lower() in self.PRE_INTENS:
                plus_next = True
                i += 1
                continue

            # Number + unit (e.g., 5 minutes -> 5-MINUTE)
            if tok.like_num and i + 1 < N:
                nxt = doc[i + 1]
                nxt_lem = nxt.lemma_.lower()
                if nxt_lem in self.UNIT_MAP:
                    mapped.append(f"{tok.text}-{self.UNIT_MAP[nxt_lem]}")
                    # inject any possessives scheduled for this head index (rare for numbers, but safe)
                    if i in pending_poss_after:
                        mapped.extend(pending_poss_after.pop(i))
                    i += 2
                    continue

            # Perfect aspect: have/has/had + VBN -> FINISH (keep participle)
            if tok.lemma_.lower() == "have" and i + 1 < N and doc[i + 1].tag_ == "VBN":
                mapped.append("FINISH")
                if i in pending_poss_after:
                    mapped.extend(pending_poss_after.pop(i))
                i += 1  # skip mapping 'have'; process participle in next loop
                continue

            # AM/PM after a number -> uppercase marker
            if tok.lemma_.lower() in self.AM_PM:
                if mapped and re.fullmatch(r"\d+", mapped[-1]):
                    mapped.append(tok.lemma_.upper())
                if i in pending_poss_after:
                    mapped.extend(pending_poss_after.pop(i))
                i += 1
                continue

            # Regular mapping
            g = self.map_token_spacy(tok)
            if g:
                if plus_next:
                    g = g + "++"
                    plus_next = False
                mapped.append(g)

            # Emit any possessives scheduled for this head token
            if i in pending_poss_after:
                mapped.extend(pending_poss_after.pop(i))

            i += 1

        # Safety: flush any leftover scheduled possessives (very rare)
        if pending_poss_after:
            for _, poss_list in sorted(pending_poss_after.items()):
                mapped.extend(poss_list)
            pending_poss_after.clear()

        # 6) ASL structure rules
        mapped = self.front_time_topic(mapped)

        # Fix English inversion for YN questions (CAN you...? -> YOU CAN...?)
        mapped = self.reorder_modal_inversion(mapped, qtype)

        if neg and "NOT" not in mapped:
            mapped = self.place_negation(mapped)

        sent_nmm = {"brows": None, "head": "shake" if neg else None, "qtype": None}
        if qtype == "WH":
            mapped = self.move_wh_to_end(mapped)
            mapped.append("?")
            sent_nmm["brows"] = "furrow"
            sent_nmm["qtype"] = "WH"
        elif qtype == "YN":
            mapped.append("?")
            sent_nmm["brows"] = "raise"
            sent_nmm["qtype"] = "YN"

        mapped = self.collapse_duplicates(mapped)
        gloss_str = " ".join([t for t in mapped if t]).strip()

        if self.t5_enabled:
            gloss_str = self._ml_refine_gloss(text, gloss_str, qtype)

        return GlossResult(gloss_tokens=mapped, gloss=gloss_str, sentence_nmm=sent_nmm)

