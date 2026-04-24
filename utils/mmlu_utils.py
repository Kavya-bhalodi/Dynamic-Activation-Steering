import json
import os
from typing import Dict, List, Any
from datasets import load_dataset


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    prompt = f"{question}\n"
    for idx, choice in enumerate(choices):
        prompt += f"{CHOICE_LABELS[idx]}. {choice}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D)."
    return prompt


def extract_answer_letter(response: str) -> Optional[str]:
    resp = response.strip().upper()
    if resp in CHOICE_LABELS:
        return resp
    patterns = [
        "\\b([A-D])\\b\\)?\\.?\\s*$",
        "(?:ANSWER|OPTION)\\s*(?:IS|:)?\\s*\\(?([A-D])\\)?",
        "^\\(?([A-D])\\)?[\\.\\,\\:\\s]",
        "\\(([A-D])\\)",
    ]
    for pat in patterns:
        match = re.search(pat, resp, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search("\\b([A-D])\\b", resp)
    if match:
        return match.group(1).upper()
    return None


def _normalize_mmlu_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if "question" not in item or "choices" not in item or "answer" not in item:
        raise ValueError("Each MMLU item must contain question, choices, answer fields")
    choices = item["choices"]
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError("Each MMLU item must contain exactly 4 choices")
    answer = item["answer"]
    if isinstance(answer, int):
        answer_idx = answer
    elif isinstance(answer, str):
        answer = answer.strip().upper()
        if answer in CHOICE_LABELS:
            answer_idx = CHOICE_LABELS.index(answer)
        elif answer.isdigit():
            answer_idx = int(answer)
        else:
            raise ValueError(f"Unrecognized answer value: {answer}")
    else:
        raise ValueError(f"Unsupported answer type: {type(answer).__name__}")
    if answer_idx < 0 or answer_idx > 3:
        raise ValueError(f"Answer index out of range: {answer_idx}")
    return {
        "question": str(item["question"]),
        "subject": str(item.get("subject", "unknown_subject")),
        "choices": [str(c) for c in choices],
        "answer": answer_idx,
    }


def load_mmlu_from_json(mmlu_path: str, max_samples: Optional[int] = None):
    if not os.path.exists(mmlu_path):
        raise FileNotFoundError(
            f"MMLU file not found: {mmlu_path}. Place balanced-mmlu-questions-across-subjects.json in phase11_data/mmlu/."
        )
    with open(mmlu_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            data = raw["data"]
        elif "questions" in raw and isinstance(raw["questions"], list):
            data = raw["questions"]
        else:
            raise ValueError("JSON dict must include list under 'data' or 'questions'")
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError(f"Expected JSON list/dict, got {type(raw).__name__}")
    norm_data = [_normalize_mmlu_item(item) for item in data]
    if max_samples is not None and max_samples > 0:
        norm_data = norm_data[:max_samples]
    subjects = defaultdict(list)
    for item in norm_data:
        subjects[item["subject"]].append(item)
    print(f"Loaded {len(norm_data)} MMLU questions across {len(subjects)} subjects")
    return (norm_data, dict(subjects))


def load_mmlu_from_hf(
    dataset_name: str,
    config_name: str,
    split_name: str,
    max_samples: Optional[int] = None,
):
    if load_dataset is None:
        raise RuntimeError(
            "datasets package is not installed. Install with: pip install datasets"
        )
    print(
        f"Loading MMLU from Hugging Face: dataset={dataset_name}, config={config_name}, split={split_name}"
    )
    try:
        ds = load_dataset(dataset_name, config_name, split=split_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF MMLU dataset {dataset_name}/{config_name}:{split_name}: {exc}"
        )
    rows = []
    for item in ds:
        row = {
            "question": item.get("question", ""),
            "subject": item.get("subject", config_name),
            "choices": item.get("choices", []),
            "answer": item.get("answer", None),
        }
        rows.append(_normalize_mmlu_item(row))
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    subjects = defaultdict(list)
    for item in rows:
        subjects[item["subject"]].append(item)
    print(f"Loaded {len(rows)} MMLU questions across {len(subjects)} subjects (HF)")
    return (rows, dict(subjects))


def _save_json(path: str, payload: Dict[str, Any]):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
