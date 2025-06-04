import json
from pathlib import Path
import torch
from transformers import BartForConditionalGeneration
from amr_runtime.model_interface.tokenization_bart import AMRBartTokenizer

# ─── 1. EDIT THESE TWO LINES ────────────────────────────────────────────────
INPUT_FILE  = "preprocessed_for_amr.json"     # your file with sentences (JSON list)
OUTPUT_FILE = "amr_graphs.json"    # where AMRs will be saved
# ────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sentences(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON *list* in {path}, got {type(data)}")
    return data


def save_json(obj, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def init_amrbart():
    print("[INFO] Loading AMRBART model…")
    tok = AMRBartTokenizer.from_pretrained(MODEL_NAME)
    mdl = (
        BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        .to(DEVICE)
        .eval()
    )
    return tok, mdl


def sentence_to_amr(sentence: str, tok, mdl) -> str:
    inputs = tok(sentence, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = mdl.generate(**inputs, max_length=256, num_beams=5)
    return tok.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def sentences_to_amr(sentences, tok, mdl, batch_size=8):
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            ids = mdl.generate(**inputs, max_length=256, num_beams=5)
        amrs = tok.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.extend(amrs)
    return results


def main() -> None:
    in_path  = Path(INPUT_FILE)
    out_path = Path(OUTPUT_FILE)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    sentences = load_sentences(in_path)
    if not sentences:
        raise ValueError("😕 The input file is empty!")

    tok, mdl = init_amrbart()

    results = sentences_to_amr(sentences, tok, mdl)

    save_json(results, out_path)
    print(f"[INFO] ✅ Saved {len(results)} AMR graphs to {out_path}")


if __name__ == "__main__":
    main()
