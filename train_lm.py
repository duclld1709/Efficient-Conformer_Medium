import argparse
import logging
import os
import subprocess
import sys
import kenlm_utils

import sentencepiece as spm

"""
NeMo's beam search decoders only support char-level encodings. In order to make it work with BPE-level encodings, we
use a trick to encode the sub-word tokens of the training data as unicode characters and train a char-level KenLM. 
TOKEN_OFFSET is the offset in the unicode table to be used to encode the BPE sub-words. This encoding scheme reduces 
the required memory significantly, and the LM and its binary blob format require less storage space.
"""
TOKEN_OFFSET = 100
CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512

def main():
    parser = argparse.ArgumentParser(
        description='Train an N-gram language model with KenLM to be used with beam search decoder of ASR models.'
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/lm_corpus_all.txt",
        help="Path to the training file, it can be a text file or JSON manifest",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="datasets/LibriSpeech/LibriSpeech_bpe_256.model", help="Tokenizer path"
    )
    parser.add_argument(
        "--kenlm_model_file", type=str, default="data/6gram_lm_corpus.binary", help="The path to store the KenLM binary model file"
    )
    parser.add_argument("--ngram_length", type=int, default=6, help="The order of N-gram LM")
    parser.add_argument("--kenlm_bin_path", type=str, default="kenlm/build/bin/", help="The path to the bin folder of KenLM")
    parser.add_argument(
        "--do_lowercase", action='store_true', help="Whether to apply lower case conversion on the training text"
    )
    args = parser.parse_args()

    """ TOKENIZER SETUP """

    encoding_level = "subword"
    tokenizer = spm.SentencePieceProcessor(args.tokenizer_path)

    """ DATASET SETUP """
    logging.info(f"Encoding the train file '{args.train_file}' ...")
    dataset = kenlm_utils.read_train_file(args.train_file, lowercase=args.do_lowercase)
    encoded_train_file = f"{args.kenlm_model_file}.tmp.txt"
    if encoding_level == "subword":
        kenlm_utils.tokenize_text(
            dataset,
            tokenizer,
            path=encoded_train_file,
            chunk_size=CHUNK_SIZE,
            buffer_size=CHUNK_BUFFER_SIZE,
            token_offset=TOKEN_OFFSET,
        )
        # --discount_fallback is needed for training KenLM for BPE-based models
        discount_arg = "--discount_fallback"
    else:
        with open(encoded_train_file, 'w', encoding='utf-8') as f:
            for line in dataset:
                f.write(f"{line}\n")

        discount_arg = ""

    # del model

    arpa_file = f"{args.kenlm_model_file}.tmp.arpa"
    """ LMPLZ ARGUMENT SETUP """
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, 'lmplz'),
        "-o",
        f"{args.ngram_length}",
        "--text",
        encoded_train_file,
        "--arpa",
        arpa_file,
        discount_arg,
    ]

    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)
    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")
    """ BINARY BUILD """
    logging.info(f"Running binary_build command \n\n{' '.join(kenlm_args)}\n\n")
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "build_binary"),
        "trie",
        arpa_file,
        args.kenlm_model_file,
    ]
    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")

    os.remove(encoded_train_file)
    logging.info(f"Deleted the temporary encoded training file '{encoded_train_file}'.")
    os.remove(arpa_file)
    logging.info(f"Deleted the arpa file '{arpa_file}'.")


if __name__ == '__main__':
    main()