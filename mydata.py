import os
import sys
import torch
import datasets
import re
from huggingface_hub import login, HfApi, HfFolder
import datasets


# Dictionaries to convert numbers to words and vice versa
NUM_TO_WORD = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "dic",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
    "10": "mười",
}
WORD_TO_NUM = {v: k for k, v in NUM_TO_WORD.items()}  # Reverse mapping


def replace_numbers_with_words(text):
    """Replace numeric digits with their Vietnamese words."""
    for num, word in NUM_TO_WORD.items():
        text = re.sub(rf"\b{num}\b", word, text)  # Ensure whole-word match
    return text


def replace_words_with_numbers(text):
    """Replace Vietnamese words representing numbers with digits."""
    for word, num in WORD_TO_NUM.items():
        text = re.sub(rf"\b{word}\b", num, text)  # Ensure whole-word match
    return text


class DataMd:
    def __init__(self) -> None:
        pass

    def normalize_text(text, words_to_numbers=False):
        text = text.lower()
        if words_to_numbers:
            text = replace_words_with_numbers(text)
        else:
            text = replace_numbers_with_words(text)
        # Step 4: Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Step 5: Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def map_dic_enc(
        dic, data_mapper, rm_answer_key: str = None, max_length=42, fp16=True
    ):
        """dic: {images: , text: , suffix: }"""
        pdic = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": max_length,
        }
        pdic.update(dic)
        if rm_answer_key is not None:
            pdic[rm_answer_key] = None
            fp16 = False

        encoding = data_mapper(**pdic)
        for k, v in encoding.items():
            encoding[k] = v.squeeze()  # remove all shape of 1, no efffect
        if fp16:
            return encoding.to(torch.float16)
        return encoding

    def docode_ids_text(ids, data_mapper, batch, skip_special_tokens=False):
        if batch:
            return data_mapper.batch_decode(
                ids, skip_special_tokens=skip_special_tokens
            )
        return data_mapper.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_torch_dataset(ds, map_fn, batched=True, remove_columns=True):
        if remove_columns:
            remove_columns = ds.column_names
        else:
            remove_columns = []
        torch_ds = ds.map(map_fn, batched=batched, remove_columns=remove_columns)
        torch_ds.set_format("torch")
        return torch_ds

    def push_dataset_to_hub(ds, repo_name, split, hf_token):
        """Pushes to a Hugging Face Hub repository."""
        # Create the repository (if not exist)
        api = HfApi()
        api.create_repo(
            repo_id=repo_name,
            private=False,
            repo_type="dataset",
            exist_ok=True,  # Allow skipping creation if repo already exists
            token=hf_token,  # Pass the Hugging Face token
        )
        ds.push_to_hub(repo_id=repo_name, split=split)


if __name__ == "__main__":
    root = os.getcwd()
    print(root)
