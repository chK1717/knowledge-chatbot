from __future__ import annotations

import hashlib
import importlib
import os
import pickle
import re
import shutil
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from pyxlsb import open_workbook
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# CONFIGURATION

FILE_PATH  = Path("Extraction_Synthese_Modélisation_SDV5.xlsb")
SHEET_NAME = "Modeling_Synthesis_SDV5"

NUM_COLS      = ["Min", "Max", "Moy", "mln", "sln", "0.01", "0.5", "0.99"]
NUMS_CONTEXTE = ["Min", "Max", "Moy"]
TEXT_COLS     = [
    "Critere", "Zone", "Periode", "Unite",
    "Segment / Energie", "Chargement", "Vehicule", "Loi Stastique",
]

ALPHA_DEFAULT           = 0.6
SCORE_MIN_DEFAULT       = 0.45
TOP_K_DEFAULT           = 7
LOCAL_EMBED_DIR         = Path("modeles/intfloat_multilingual-e5-base")
REMOTE_EMBED_ID         = "intfloat/multilingual-e5-base"
COLLECTION_NAME_DEFAULT = f"sdv_{REMOTE_EMBED_ID.split('/')[-1].replace('-', '_')}"
CHROMA_DIR              = Path("sdv_chromadb")
CACHE_BM25              = Path("sdv_bm25_cpu.pkl")
LLM_MODEL_PATH          = Path("modeles/qwen/qwen2.5-3b-instruct-q8_0-005.gguf")

# SYSTEM PROMPT
SYSTEM_PROMPT = (
    'Tu es un assistant expert donnees vehicules EV/PHEV Stellantis.\n'
    'REGLES ABSOLUES :\n'
    '1. Reponds OBLIGATOIREMENT dans la meme langue que la question.\n'
    '2. Utilise UNIQUEMENT les valeurs du contexte fourni.\n'
    '3. INTERDIT : inventer, calculer, estimer ou interpreter.\n'
    '4. Zone : recopie EXACTEMENT la valeur Zone du contexte.\n'
    '   Si Zone est vide ou "-" ecrire "Non specifiee".\n'
    '   Si Zone a une valeur recopie cette valeur exactement.\n'
    '5. Reponds directement a la question posee.\n'
    '6. Si valeur absente : Non disponible.\n'
    '7. Sois concis et precis. Maximum 5 lignes.'
)


# DATACLASS
@dataclass
class AppResources:
    df            : pd.DataFrame
    row_indices   : list[int]
    doc_ids       : list[str]
    doc_id_to_idx : dict[str, int]
    bm25_index    : BM25Okapi
    embedder      : SentenceTransformer
    collection    : Any
    llm           : Any | None
    llm_available : bool
    llm_error     : str | None


# CHARGEMENT DONNEES
def _resolve_xlsb_path() -> Path:
    if FILE_PATH.exists():
        return FILE_PATH
    matches = sorted(Path(".").glob("Extraction_Synthese_*SDV5.xlsb"))
    if matches:
        return matches[0]
    return FILE_PATH


def read_xlsb(file_path: Path, sheet_name: str) -> pd.DataFrame:
    def _read(path: Path) -> pd.DataFrame:
        rows = []
        with open_workbook(str(path)) as wb:
            with wb.get_sheet(sheet_name) as sheet:
                for row in sheet.rows():
                    rows.append([cell.v for cell in row])
        headers = rows[0]
        out = pd.DataFrame(rows[1:], columns=headers)
        out.dropna(how="all", inplace=True)
        out.reset_index(drop=True, inplace=True)
        for col in NUM_COLS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    try:
        return _read(file_path)
    except PermissionError:
        tmp = Path(tempfile.gettempdir()) / f"sdv_copy_{int(time.time())}.xlsb"
        try:
            shutil.copyfile(file_path, tmp)
            return _read(tmp)
        except Exception as e:
            raise RuntimeError(
                f"Impossible de lire {file_path.name}. "
                "Fermez le fichier dans Excel puis reessayez."
            ) from e
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass


def row_to_text(row: pd.Series) -> str:
    parts: list[str] = []
    critere = str(row.get("Critere", "") or "").strip()
    if critere and critere not in ["-", "None"]:
        parts += [critere, critere, critere]
    for col in ["Zone", "Periode", "Unite", "Segment / Energie",
                "Chargement", "Vehicule", "Loi Stastique"]:
        val = str(row.get(col, "") or "").strip()
        if val and val not in ["-", "None"]:
            parts.append(f"{col}: {val}")
    for col in ["Min", "Max", "Moy"]:
        if col in row.index and pd.notna(row[col]):
            try:
                parts.append(f"{col}: {round(float(row[col]), 4)}")
            except Exception:
                pass
    return " ".join(parts)


def row_to_bm25(row: pd.Series) -> list[str]:
    tokens: list[str] = []
    for col in TEXT_COLS:
        val = str(row.get(col, "") or "").strip()
        if val and val not in ["-", "None"]:
            toks = (val.lower()
                    .replace("/", " ").replace("_", " ")
                    .replace(".", " ").replace("?", " ")
                    .replace(":", " ").split())
            tokens.extend(toks)
    return tokens


@st.cache_resource(show_spinner=False)
def load_resources(collection_name: str) -> AppResources:
    xlsb_path = _resolve_xlsb_path()
    if not xlsb_path.exists():
        raise FileNotFoundError("XLSB introuvable.")

    df = read_xlsb(xlsb_path, SHEET_NAME)

    texts: list[str]           = []
    bm25_docs: list[list[str]] = []
    row_indices: list[int]     = []
    doc_ids: list[str]         = []
    seen_hashes: set[str]      = set()

    for idx, row in df.iterrows():
        text = row_to_text(row)
        if len(text.strip()) < 20:
            continue
        h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        texts.append(text)
        bm25_docs.append(row_to_bm25(row))
        row_indices.append(int(idx))
        doc_ids.append(f"doc_{idx}")

    embedder = None
    for model_name in [str(LOCAL_EMBED_DIR), REMOTE_EMBED_ID]:
        try:
            kwargs: dict[str, Any] = {"device": "cpu"}
            if model_name == str(LOCAL_EMBED_DIR):
                kwargs["local_files_only"] = True
            embedder = SentenceTransformer(model_name, **kwargs)
            break
        except Exception:
            continue
    if embedder is None:
        raise RuntimeError("Embedder introuvable.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing      = [c.name for c in chroma_client.list_collections()]
    collection    = None
    if collection_name in existing:
        collection = chroma_client.get_collection(collection_name)
        if collection.count() == 0:
            chroma_client.delete_collection(collection_name)
            collection = None
    if collection is None:
        collection = chroma_client.create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})
        for s in range(0, len(texts), 500):
            e    = min(s + 500, len(texts))
            embs = embedder.encode(
                texts[s:e], batch_size=32,
                normalize_embeddings=True, show_progress_bar=False,
            ).tolist()
            collection.add(ids=doc_ids[s:e], embeddings=embs, documents=texts[s:e])

    if CACHE_BM25.exists():
        with CACHE_BM25.open("rb") as f:
            bm25_index = pickle.load(f)
    else:
        bm25_index = BM25Okapi(bm25_docs)
        with CACHE_BM25.open("wb") as f:
            pickle.dump(bm25_index, f)

    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    llm           = None
    llm_available = False
    llm_error     = None
    if LLM_MODEL_PATH.exists():
        try:
            Llama = importlib.import_module("llama_cpp").Llama
            llm   = Llama(
                model_path      = str(LLM_MODEL_PATH),
                n_threads       = 6,
                n_threads_batch = 6,
                n_ctx           = 768,
                n_batch         = 1024,
                use_mmap        = True,
                use_mlock       = True,
                verbose         = False,
                logits_all      = False,
            )
            llm_available = True
        except Exception as exc:
            llm_error = str(exc)
    else:
        llm_error = f"GGUF introuvable: {LLM_MODEL_PATH}"

    return AppResources(
        df=df, row_indices=row_indices, doc_ids=doc_ids,
        doc_id_to_idx=doc_id_to_idx, bm25_index=bm25_index,
        embedder=embedder, collection=collection,
        llm=llm, llm_available=llm_available, llm_error=llm_error,
    )


# RECHERCHE HYBRIDE
def hybrid_search(
    question: str, resources: AppResources,
    top_k: int, alpha: float, score_min: float,
) -> list[dict[str, Any]]:
    n_docs     = len(resources.doc_ids)
    chroma_map: dict[int, float] = {}

    coll_size = resources.collection.count() if resources.collection else 0
    if coll_size > 0:
        q_emb    = resources.embedder.encode([question], normalize_embeddings=True).tolist()
        chroma_r = resources.collection.query(
            query_embeddings=q_emb,
            n_results=min(top_k * 10, coll_size),
        )
        if chroma_r["ids"] and chroma_r["ids"][0]:
            for doc_id, dist in zip(chroma_r["ids"][0], chroma_r["distances"][0]):
                idx = resources.doc_id_to_idx.get(doc_id, -1)
                if idx >= 0:
                    chroma_map[idx] = 1.0 - float(dist)

    tokens    = question.lower().replace("/", " ").replace("?", " ").replace(":", " ").split()
    bm25_raw  = resources.bm25_index.get_scores(tokens)
    bm25_max  = float(np.max(bm25_raw)) if float(np.max(bm25_raw)) > 0 else 1.0
    bm25_norm = bm25_raw / bm25_max

    candidates = set(chroma_map.keys()) | set(np.argsort(bm25_norm)[-top_k * 10:])
    results: list[dict[str, Any]] = []
    for idx in candidates:
        if idx < 0 or idx >= n_docs:
            continue
        c = chroma_map.get(int(idx), 0.0)
        b = float(bm25_norm[int(idx)])
        s = alpha * c + (1 - alpha) * b
        if s >= score_min:
            results.append({
                "idx"         : int(idx),
                "score_final" : round(float(s), 4),
                "score_chroma": round(float(c), 4),
                "score_bm25"  : round(float(b), 4),
                "df_row"      : resources.df.iloc[resources.row_indices[int(idx)]],
            })
    results.sort(key=lambda x: x["score_final"], reverse=True)
    return results[:top_k]


# LLM
def ask_llm(
    resources: AppResources, prompt: str, max_tokens: int = 80
) -> tuple[str, float]:
    if not resources.llm_available or resources.llm is None:
        return "LLM indisponible", 0.0
    try:
        t0  = time.time()
        out = resources.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens     = max_tokens,
            temperature    = 0.05,
            repeat_penalty = 1.15,
            stop=[
                "</s>", "[INST]", "---", "CONTEXTE", "STOP.",
                "There is no", "Il n'y a pas",
                "Note :", "Remarque", "However", "Cependant",
                "\nNon disponible", "\nNon disponible.",
            ],
        )
        return out["choices"][0]["message"]["content"].strip(), round(time.time()-t0, 2)
    except Exception as exc:
        return f"Erreur LLM: {exc}", 0.0


def build_prompt(question: str, docs: list[dict[str, Any]]) -> str:
    parts = []
    for rank, item in enumerate(docs, start=1):
        row   = item["df_row"]
        lines = [f"--- Donnee #{rank} ---"]
        for col in ["Critere", "Zone", "Periode", "Unite",
                    "Segment / Energie", "Chargement", "Loi Stastique"]:
            if col in row.index and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val and val not in ["-", "None"]:
                    lines.append(f"{col}: {val}")
        for col in NUMS_CONTEXTE:
            if col in row.index and pd.notna(row[col]):
                try:
                    lines.append(f"{col}: {round(float(row[col]), 4)}")
                except Exception:
                    pass
        parts.append("\n".join(lines))

    context = "\n\n".join(parts)
    return (
        f"CONTEXTE:\n\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"INSTRUCTION : Reponds en utilisant UNIQUEMENT les valeurs du contexte.\n"
        f"Recopie TOUTES les valeurs : Critere, Zone, Periode, Loi, Min, Max, Moy.\n"
        f"ZERO calcul. ZERO invention.\n"
    )


# UTILITAIRES
def _normalize_text(value: str) -> str:
    txt = unicodedata.normalize("NFKD", value)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    return txt.lower().strip()


def _has_numeric_values(row: pd.Series) -> bool:
    for col in ["Min", "Max", "Moy"]:
        if col in row.index and pd.notna(row[col]):
            try:
                if float(row[col]) != 0:
                    return True
            except Exception:
                pass
    return False


def _doc_priority(item: dict[str, Any]) -> tuple[int, int, int, float]:
    row     = item["df_row"]
    loi     = str(row.get("Loi Stastique", "") or "").lower()
    periode = str(row.get("Periode",       "") or "").lower()

    loi_rank = 0 if "log" in loi else (2 if loi in {"", "to add", "-"} else 1)
    num_rank = 0 if _has_numeric_values(row) else 1

    if any(p in periode for p in ["year", "ans", "km"]):
        per_rank = 0
    elif periode in ["day", "min", "sec", "h", ""]:
        per_rank = 2
    else:
        per_rank = 1

    return (num_rank, loi_rank, per_rank, -float(item.get("score_final", 0.0)))


def select_docs_for_llm(
    question: str, results: list[dict[str, Any]], llm_docs: int
) -> list[dict[str, Any]]:
    return sorted(results, key=_doc_priority)[: max(1, llm_docs)]


def _num_or_na(row: pd.Series, col: str) -> str:
    if col in row.index and pd.notna(row[col]):
        try:
            return str(round(float(row[col]), 4))
        except Exception:
            pass
    return "Non disponible"


def _build_zone_unavailable_answer(zone_demandee: str, row: pd.Series) -> str:
    critere = str(row.get("Critere", "") or "").strip() or "Non disponible"
    zone    = str(row.get("Zone", "") or "").strip() or "Non disponible"
    unite   = str(row.get("Unite", "") or "").strip()
    loi     = str(row.get("Loi Stastique", "") or "").strip() or "Non disponible"
    suffix  = f" {unite}" if unite else ""
    min_v   = _num_or_na(row, "Min")
    max_v   = _num_or_na(row, "Max")
    moy_v   = _num_or_na(row, "Moy")

    return (
        f"La zone **{zone_demandee}** n'existe pas dans les donnees.\n\n"
        f"Voici les donnees disponibles pour **{zone}** :\n\n"
        f"Critere : {critere}\n"
        f"Zone    : {zone}\n"
        f"Unite   : {unite or 'Non disponible'}\n"
        f"Loi     : {loi}\n"
        f"Min     : {min_v}{suffix if min_v != 'Non disponible' else ''}\n"
        f"Max     : {max_v}{suffix if max_v != 'Non disponible' else ''}\n"
        f"Moy     : {moy_v}{suffix if moy_v != 'Non disponible' else ''}"
    )


def _build_grounded_answer(row: pd.Series) -> str:
    critere = str(row.get("Critere", "") or "").strip() or "Non disponible"
    zone    = str(row.get("Zone", "") or "").strip() or "Non disponible"
    unite   = str(row.get("Unite", "") or "").strip()
    loi     = str(row.get("Loi Stastique", "") or "").strip() or "Non disponible"
    suffix  = f" {unite}" if unite else ""
    min_v   = _num_or_na(row, "Min")
    max_v   = _num_or_na(row, "Max")
    moy_v   = _num_or_na(row, "Moy")

    return (
        f"Critere : {critere}\n"
        f"Zone    : {zone}\n"
        f"Unite   : {unite or 'Non disponible'}\n"
        f"Loi     : {loi}\n"
        f"Min     : {min_v}{suffix if min_v != 'Non disponible' else ''}\n"
        f"Max     : {max_v}{suffix if max_v != 'Non disponible' else ''}\n"
        f"Moy     : {moy_v}{suffix if moy_v != 'Non disponible' else ''}"
    )


def _pick_best_row(resources: AppResources, results: list[dict[str, Any]]) -> pd.Series:
    if not results:
        return pd.Series(dtype=object)
    best_docs = select_docs_for_llm("", results, 1)
    return best_docs[0]["df_row"] if best_docs else results[0]["df_row"]


def _extract_requested_zone(question: str, known_zones: list[str]) -> str | None:
    q_lc = question.lower()
    for z in known_zones:
        if z.lower() in q_lc:
            return z
    m = re.search(
        r"\b(?:en|au|aux|a|dans)\s+([^\?\!\.,:;]+)",
        question, flags=re.IGNORECASE
    )
    if not m:
        return None
    candidate = m.group(1).strip()
    candidate = re.split(
        r"\b(?:pour|avec|sur|et|ou)\b", candidate,
        maxsplit=1, flags=re.IGNORECASE
    )[0].strip().strip(" -_.,:;!?")
    return candidate or None


def _extract_zone_after_preposition(question: str) -> str | None:
    m = re.search(
        r"\b(?:en|au|aux|a|dans)\s+([^\?\!\.,:;]+)",
        question, flags=re.IGNORECASE
    )
    if not m:
        return None
    candidate = m.group(1).strip()
    candidate = re.split(
        r"\b(?:pour|avec|sur|et|ou)\b", candidate,
        maxsplit=1, flags=re.IGNORECASE
    )[0].strip().strip(" -_.,:;!?")
    return candidate or None


# MEMOIRE CONVERSATIONNELLE
def _is_followup(question: str) -> bool:
    q = question.lower().strip()
    mots_suivi = (
        "et ", "mais ", "compare", "versus", "vs ",
        "et pour", "et en ", "et avec ", "et la ",
        "et le ", "et les ", "et du ", "et des ",
        "aussi", "pareil", "et sur", "idem",
    )
    return q.startswith(mots_suivi)


def contextualize_question(
    question: str, chat_messages: list[dict[str, Any]]
) -> str:
    if not _is_followup(question):
        return question

    msgs = list(chat_messages)
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if msg.get("role") != "user":
            continue
        next_assistant = None
        for j in range(i + 1, len(msgs)):
            if msgs[j].get("role") == "assistant":
                next_assistant = msgs[j]
                break
        if next_assistant:
            content = str(next_assistant.get("content", "")).lower()
            if any(w in content for w in [
                "pertinence faible", "reformulez",
                "introuvable", "non disponible"
            ]):
                continue
        return f"{str(msg['content']).strip()} | Suivi: {question.strip()}"

    return question


# AFFICHAGE
def render_compact_sources(
    results: list[dict[str, Any]], max_items: int = 3
) -> None:
    if not results:
        st.caption("Aucune source.")
        return
    displayed = 0
    for item in results:
        row     = item["df_row"]
        critere = str(row.get("Critere", "") or "").strip()
        if not critere or critere in ["-", "None"]:
            continue
        zone    = str(row.get("Zone", "") or "").strip()
        periode = str(row.get("Periode", "") or "").strip()
        st.markdown(
            f"{displayed+1}. score={item['score_final']:.3f}"
            f" | {critere[:60]} | {zone} | {periode}"
        )
        displayed += 1
        if displayed >= max_items:
            break


# MAIN
def main() -> None:
    st.set_page_config(page_title="SmartBot SDV", layout="wide")
    st.title("SmartBot SDV — Stellantis")
    st.caption("RAG hybride ChromaDB + BM25 + Qwen2.5-3B Q8 (CPU)")

    with st.sidebar:
        st.header("Parametres")
        collection_name = st.text_input(
            "Collection Chroma", value=COLLECTION_NAME_DEFAULT)
        top_k      = st.slider("Top K", 1, 20, TOP_K_DEFAULT)
        alpha      = st.slider("Alpha (Chroma vs BM25)", 0.0, 1.0,
                               ALPHA_DEFAULT, step=0.05)
        score_min  = st.slider("Score min", 0.0, 1.0,
                               SCORE_MIN_DEFAULT, step=0.01)
        llm_docs   = st.slider("Docs injectes LLM", 1, 5, 1)
        max_tokens = st.slider("Max new tokens", 32, 512, 80, step=16)
        st.caption("Conseil : garder 1 doc + 80 tokens.")
        show_sources = st.checkbox("Afficher sources", value=True)
        show_prompt  = st.checkbox("Afficher prompt LLM", value=False)
        force_reload = st.button("Recharger ressources")
        clear_chat   = st.button("Effacer conversation")

    if force_reload:
        load_resources.clear()

    try:
        with st.spinner("Chargement des ressources..."):
            resources = load_resources(collection_name)
    except Exception as exc:
        msg = str(exc)
        st.error("Echec de chargement.")
        if "Permission" in msg or "Impossible" in msg:
            st.info("Fermez le fichier Excel puis cliquez Recharger.")
        st.code(msg)
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes", f"{len(resources.df):,}")
    c2.metric("Docs indexes", f"{len(resources.doc_ids):,}")
    c3.metric("LLM", "Actif" if resources.llm_available else "Inactif")

    if not resources.llm_available and resources.llm_error:
        st.info(f"LLM indisponible : {resources.llm_error}")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if clear_chat:
        st.session_state.chat_messages = []

    if not st.session_state.chat_messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Bonjour ! Je suis votre assistant SDV Stellantis. "
                "Posez une question sur les donnees vehicules "
                "(ex: *Duree recharge rapide Europe PHEV 3 ans*)."
            )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if show_sources:
                    with st.expander("Sources", expanded=False):
                        render_compact_sources(msg.get("sources") or [])
                if show_prompt and msg.get("prompt"):
                    with st.expander("Prompt LLM", expanded=False):
                        st.code(msg["prompt"], language="text")

    question = st.chat_input("Posez votre question...")
    if not question:
        return

    effective_question = contextualize_question(
        question, st.session_state.chat_messages
    )

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Recherche en cours..."):
            results = hybrid_search(
                question  = effective_question.strip(),
                resources = resources,
                top_k     = top_k,
                alpha     = alpha,
                score_min = score_min,
            )

        if not results:
            answer = "Aucune donnee pertinente trouvee. Reformulez la question."
            st.markdown(answer)
            st.session_state.chat_messages.append(
                {"role":"assistant","content":answer,"sources":[],"prompt":""}
            )
            return

        max_score = results[0]["score_final"]

        # Bloquer si score trop faible
        if max_score < 0.50:
            answer = (
                f"**Donnee non disponible** (score={max_score:.2f})\n\n"
                f"Critere le plus proche :\n"
                f"> {str(results[0]['df_row'].get('Critere',''))[:100]}"
            )
            st.warning(answer)
            st.session_state.chat_messages.append(
                {"role":"assistant","content":answer,"sources":results,"prompt":""}
            )
            return

        elif max_score < 0.75:
            answer = (
                f"**Pertinence faible** (score={max_score:.2f})\n\n"
                f"Critere le plus proche :\n"
                f"> {str(results[0]['df_row'].get('Critere',''))[:100]}\n\n"
                f"*Reformulez avec des mots plus precis.*"
            )
            st.warning(answer)
            st.session_state.chat_messages.append(
                {"role":"assistant","content":answer,"sources":results,"prompt":""}
            )
            return

        # Donnees qualitatives sans Min/Max/Moy
        best_numeric = [r for r in results if _has_numeric_values(r["df_row"])]
        if not best_numeric:
            if max_score > 0.85:
                answer = (
                    "**Critere trouve** mais donnees matricielles/distribution.\n\n"
                    "Ce critere ne contient pas de valeurs Min/Max/Moy scalaires.\n\n"
                    "Consultez directement le fichier Excel."
                )
            else:
                answer = (
                    "**Donnees qualitatives** — pas de Min/Max/Moy disponibles.\n\n"
                    "Reformulez avec le nom exact du critere."
                )
            st.info(answer)
            if show_sources:
                with st.expander("Sources", expanded=False):
                    render_compact_sources(results)
            st.session_state.chat_messages.append(
                {"role":"assistant","content":answer,"sources":results,"prompt":""}
            )
            return

        # Detection zone absente
        zones_trouvees: set[str] = set()
        for r in results:
            z = str(r["df_row"].get("Zone", "") or "").strip()
            if z and z not in ["-", "None", ""]:
                zones_trouvees.add(z)

        all_zones = sorted(
            zones_trouvees | {
                str(z).strip()
                for z in resources.df["Zone"].dropna().unique()
                if str(z).strip() not in ["", "-", "None"]
            },
            key=len, reverse=True,
        )

        zone_demandee      = _extract_requested_zone(question, all_zones)
        explicit_zone      = _extract_zone_after_preposition(question)
        explicit_zone_ctx  = _extract_zone_after_preposition(effective_question)
        zones_trouvees_lc  = {_normalize_text(z) for z in zones_trouvees}
        explicit_effective = explicit_zone or explicit_zone_ctx

        zone_absente  = False
        zone_demandee_label = zone_demandee or explicit_effective

        if zones_trouvees and zone_demandee_label:
            if _normalize_text(zone_demandee_label) not in zones_trouvees_lc:
                zone_absente = True

        if zone_absente:
            best = _pick_best_row(resources, results)
            st.info(
                f"**Zone '{zone_demandee_label}' non disponible.**\n\n"
                f"Zones trouvees pour ce critere : "
                f"{', '.join(sorted(zones_trouvees))}\n\n"
                f"**Reponse avec les donnees disponibles :**"
            )
            answer  = _build_zone_unavailable_answer(zone_demandee_label, best)
            elapsed = 0.0
            st.markdown(answer)
            st.caption(f"Temps generation: {elapsed}s (mode deterministe)")
            if show_sources:
                with st.expander("Sources", expanded=False):
                    render_compact_sources(results)
            st.session_state.chat_messages.append(
                {"role":"assistant","content":answer,"sources":results,"prompt":""}
            )
            return

        docs_for_llm = select_docs_for_llm(
            effective_question.strip(), results, llm_docs
        )
        prompt = build_prompt(effective_question.strip(), docs_for_llm)

        with st.spinner("Generation LLM..."):
            answer, elapsed = ask_llm(resources, prompt, max_tokens=max_tokens)

        has_values_in_answer = bool(
            re.search(r"min\s*:\s*\d", answer, flags=re.IGNORECASE)
            and re.search(r"max\s*:\s*\d", answer, flags=re.IGNORECASE)
            and re.search(r"moy\s*:\s*\d", answer, flags=re.IGNORECASE)
        )
        all_stats_missing = all(
            re.search(rf"\b{label}\s*:\s*non\s*disponible\b", answer, flags=re.IGNORECASE)
            for label in ["Min", "Max", "Moy"]
        )
        answer_too_short = len(answer.strip().split("\n")) <= 2

        if zones_trouvees and not has_values_in_answer and (all_stats_missing or answer_too_short):
            best   = _pick_best_row(resources, results)
            answer = _build_grounded_answer(best)

        st.markdown(answer)
        st.caption(f"Temps generation: {elapsed}s")

        if effective_question.strip() != question.strip():
            st.caption(f"Question contextualisee: {effective_question}")

        if show_sources:
            with st.expander("Sources", expanded=False):
                render_compact_sources(results)

        if show_prompt:
            with st.expander("Prompt LLM", expanded=False):
                st.code(prompt, language="text")

        st.session_state.chat_messages.append({
            "role"               : "assistant",
            "content"            : answer,
            "sources"            : results,
            "prompt"             : prompt,
            "effective_question" : effective_question,
        })


if __name__ == "__main__":
    main()