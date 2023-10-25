import collections
import logging
import os

from transformers import BertTokenizer


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab_cui2ind = collections.OrderedDict()
    vocab_des2ind = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        tokens = token.rstrip("\n").split("\t")
        vocab_cui2ind[tokens[0]] = index
        vocab_des2ind[tokens[1]] = index
    return vocab_cui2ind, vocab_des2ind

class CUIBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        **kwargs
    ):
        # load vocab
        self.vocab_cui2ind, self.vocab_des2ind = load_vocab(vocab_file)
        self.vocab_ind2cui = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab_cui2ind.items()])
        self.vocab_ind2des = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab_des2ind.items()])

        fn, fn_ext = os.path.splitext(vocab_file)
        vocab_file_new = fn + "_temp" + fn_ext
        # two column vocab file to one column bert vocab file
        index = 0
        with open(vocab_file_new, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab_cui2ind.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    print(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")  
                index += 1 

        # load bert's stuff
        super().__init__(
            vocab_file_new,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    def ind2cui(self, indices):
        """Converts an index (integer) in a token (str) using the vocab."""
        return [self.vocab_ind2cui[a] for a in indices]

    def ind2des(self, indices):
        """Converts an index (integer) in a token (str) using the vocab."""
        return [self.vocab_ind2des[a][:25] for a in indices]
