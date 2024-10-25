import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from MyLib.utils import ut
from MyLib._data import DataModule
import torch
from tqdm import tqdm as TQDM

nltk.download("punkt")
nltk.download("wordnet")
from pyvi import ViTokenizer
from googletrans import Translator

translator = Translator()


class EvalCPN:
    def __init__(self) -> None:
        pass

    def compute_bleu(all_decoded, all_answers):
        """Computes the average BLEU score for Vietnamese text."""
        bleu_scores = []
        for i in range(len(all_decoded)):
            # Tokenize using pyvi
            reference_tokens = [
                ViTokenizer.tokenize(all_answers[i]).split()
            ]  # Tokenize and split into words
            candidate_tokens = ViTokenizer.tokenize(
                all_decoded[i]
            ).split()  # Tokenize and split into words

            # Average the BLEU scores
            bleu_n_scores = [sentence_bleu(reference_tokens, candidate_tokens)]
            avg_bleu_score = sum(bleu_n_scores) / len(bleu_n_scores)
            bleu_scores.append(avg_bleu_score)

        average_bleu = sum(bleu_scores) / len(bleu_scores)
        return average_bleu

    def eval_step(model, testdic, data_mapper, key):
        all_decoded = []
        all_answers = testdic[key]

        model_inputs = DataModule.map_dic_enc(
            testdic,
            data_mapper,
            rm_answer_key=key,
            fp16=False,
        )
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        input_len = model_inputs["input_ids"].shape[-1]  # cut gen

        with torch.inference_mode():
            generation = model.generate(
                **model_inputs, max_new_tokens=100, do_sample=False
            )
            generation = generation[:, input_len:]  # gen
            decoded = data_mapper.batch_decode(generation, skip_special_tokens=True)
            all_decoded.extend(decoded)
            ut.mess(f"size: {len(decoded)}")
            ut.mess(f"decoded: {decoded[0]}, {all_answers[0]}")

        all_decoded_vi = [
            translator.translate(text, dest="vi").text for text in all_decoded
        ]
        all_decoded_vi_norm = [
            DataModule.normalize_text(text) for text in all_decoded_vi
        ]
        average_bleu = EvalModule.compute_bleu(all_decoded_vi_norm, all_answers)
        return average_bleu

    def eval_epoch(model, testdic, K, data_mapper, key):
        scores = []
        for i in range(0, len(testdic), K):
            scores.append(
                EvalModule.eval_step(model, testdic[i : i + K], data_mapper, key)
            )
            ut.mess(f"unit-bleu: {scores[-1]:.4f}")
        ut.mess(f"full-bleu: {sum(scores) / len(scores):.4f}")
