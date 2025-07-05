import json
from typing import Union, List
from abc import ABC
import unicodedata
from datetime import datetime
from collections import defaultdict
import pickle
from functools import partial

from normalizer import normalize as buetNormalizer


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class BaseTokenizer(ABC):
    """
    Abstract Base Class for Tokenizer
    """

    def __init__(self, vocabulary: List[str], separator: str):
        """Initialize Base Class

        Args:
            vocabulary (Union): A list string
        """
        pass

    def tokenize(self, text: str):
        """Tokenize input text

        Args:
            text (str): input text
        Returns:
            tokenized_text (list[str]): a list of token
        """
        pass


class GraphemeTokenizer:
    def __init__(
        self,
        tokenizer_class: BaseTokenizer,
        max_len: int = 64,
        separator: str = "",
        blank_token: str = "_",
        oov_token: str = "▁",
        normalize_unicode: bool = False,
        normalization_mode: str = "NFKC",
        normalizer="buetNormalizer",
        printer=print,
        bos_token: str = "_",
        eos_token: str = "_",
        add_bos_token: bool = True, 
        add_eos_token: bool = True
    ):
        """A TRIE based Sequence Tokenizer

        Args:
            tokenizer_class (BaseTokenizer)): Base Tokenizer Class.
            max_len (int, optional): Maximum length of tokenized sequence. Defaults to 64.
            separator (str, optional): 'Char' to  be used as separator. Defaults to ''.
            blank_token (str, optional): Blank token. Defaults to '_'.
            oov_token (str, optional): Out of vocab placeholder. Defaults to '▁'.
            normalize_unicode (bool, optional): Whether to normalize unicode. Defaults to False.
            normalization_mode (str, optional): unicode normalization mode['NFKC']. Defaults to 'NFKC'.
            normalizer (str, optional): which(buetNormalizer,unicode) text normalizer to use. Defaults to buetNormalizer
            printer(callable, optional): Internal stdout function, could be a logger object or print
        """

        self.vocab = [oov_token, blank_token]
        self.max_len = max_len
        self.oov_token = oov_token
        self.blank_token = blank_token

        self.tokenizer_class = tokenizer_class
        self.separator = separator
        self.tokenizer = self.tokenizer_class(
            [(idx) for idx in self.vocab], separator=self.separator
        )

        self.word2index = {token: idx for idx, token in enumerate(self.vocab)}
        self.normalize_unicode = normalize_unicode
        self.normalization_mode = normalization_mode

        self.print = printer

        self.print(f"Selected Tokenizer: {tokenizer_class.__name__}")
        self.print(f"Max Sequence Length: {max_len}")

        self.print(f"Normalize Text: {normalize_unicode}")
        self.print(f"Normalizar: {normalizer}")
        self.print(f"Normalization Mode: {normalization_mode}")

        self.out_of_vocabulary_info = defaultdict(set)
        self.frequency_counter = defaultdict(int)

        self._set_normalizer(normalizer)
        
        # added by mahmudulyeamim
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        
        self.pad_token_id = self.word2index[self.blank_token]
        self.bos_token_id = self.word2index[self.blank_token]
        self.eos_token_id = self.word2index[self.blank_token]

    def __call__(
        self,
        text: Union[str, List],
        normalize_unicode: bool = None,
        normalization_mode: bool = None,
    ):
        """Toknize input text(s) and collects all the statistics
        (out of vocabulary token, its counter and word that has out of vocabulary token)

        Args:
            text (Union[str, List]): input text
            normalize_unicode (bool, optional): whether to normalize the input. Defaults to None.
            normalization_mode (bool, optional): normalization mode. Defaults to None.

        Returns:
            dict: {
            "tokens": list(), list of tokens
            "input_ids": list(), list of token ids
            "token_len": int, number of tokens
            "contain_oov": bool, if the input if contains out of vocob token
        }
        """

        if isinstance(text, list):
            return [
                self(_text, normalize_unicode, normalization_mode) for _text in text
            ]

        normalize_unicode = (
            self.normalize_unicode if normalize_unicode is None else normalize_unicode
        )
        if normalize_unicode:
            text = self._unicode_normalizer(text, normalization_mode)

        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[: self.max_len]

        n_tokens = len(tokens)

        tokens_id = [self._get_index(text, token) for token in tokens]
        contain_oov = self.word2index[self.oov_token] in tokens_id

        return {
            "tokens": tokens,
            "input_ids": tokens_id,
            "token_len": n_tokens,
            "contain_oov": contain_oov,
        }

    def tokenize(
        self,
        text: Union[str, List],
        padding: bool = False,
        normalize_unicode: bool = None,
        normalization_mode: bool = None
    ):
        """Toknize input text(s)

        Args:
            text (Union[str, List]): input text
            normalize_unicode (bool, optional): whether to normalize the input. Defaults to None.
            normalization_mode (bool, optional): normalization mode. Defaults to None.

        Returns:
            dict: {
            "tokens": list(), list of tokens
            "input_ids": list(), list of token ids
            "token_len": int, number of tokens
            "attention_mask": list, attention mask for input ids
        }
        """
        if isinstance(text, list):
            return [
                self.tokenize(_text, padding, normalize_unicode, normalization_mode)
                for _text in text
            ]

        normalize_unicode = (
            self.normalize_unicode if normalize_unicode is None else normalize_unicode
        )
        if normalize_unicode:
            text = self._unicode_normalizer(text, normalization_mode)

        tokens = self.tokenizer.tokenize(text)
        
        # Added by mahmudulyeamim
        if self.add_bos_token and self.add_eos_token:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        elif self.add_bos_token:
            tokens = [self.bos_token] + tokens
        elif self.add_eos_token:
            tokens = tokens + [self.eos_token]
            
        tokens = tokens[: self.max_len]
        n_tokens = len(tokens)

        if padding:
            tokens = tokens + [self.blank_token] * (self.max_len - n_tokens)

        tokens_id = [
            self.word2index.get(token, self.word2index[self.oov_token])
            for token in tokens
        ]
        attention_mask = [1] * n_tokens + [0] * (len(tokens) - n_tokens)

        return {
            "tokens": tokens,
            "input_ids": tokens_id,
            "token_len": n_tokens,
            "attention_mask": attention_mask,
        }

    def add_tokens(self, vocab: list, normalize_unicode: bool = None, reset_oov=False):
        """Adds new vocabulary from a list

        Args:
            vocab (list): New vocabulary
            normalize_unicode (bool, optional): Whether to normalize unicode. Defaults to None.
            reset_oov (bool, optional): Whether to reset out_of_vocabulary_info state. Defaults to False.
        """

        normalize_unicode = (
            self.normalize_unicode if normalize_unicode is None else normalize_unicode
        )

        vocab = self._validate_tokens(vocab, normalize_unicode)

        self.vocab = self.vocab + vocab
        self.tokenizer = self.tokenizer_class(
            [(v) for v in self.vocab], separator=self.separator
        )

        self.word2index = {token: idx for idx, token in enumerate(self.vocab)}
        
        # added by mahmudulyeamim
        self.bos_token_id = self.word2index[self.bos_token]
        self.eos_token_id = self.word2index[self.eos_token]

        if reset_oov:
            self.print(
                'Warning: "out_of_vocabulary_info" will be updated as per new vocab'
            )
            self.reset_out_of_vocabulary_info(keys=vocab)
        else:
            self.print('Warning: old "out_of_vocabulary_info" will not be updated')

        self.print(
            f"update completed.[{len(vocab)}] new vocabs added. "
            + f"Current vocab count: {len(self.vocab)}"
        )

    def _validate_tokens(self, vocab, normalize_unicode: bool = False):
        if normalize_unicode:
            vocab = list(map(self._unicode_normalizer, vocab))

        vocab = sorted(list(set(vocab)))
        vocab = [v.strip() for v in vocab if v not in self.vocab]

        return vocab

    def _set_normalizer(self, type="buetNormalizer"):
        if type == "buetNormalizer":
            self.normalizer = partial(
                buetNormalizer,
                punct_replacement=None,
                url_replacement=None,
                emoji_replacement=None,
                apply_unicode_norm_last=True,
            )
        elif type == "unicode":
            self.normalizer = lambda text, mode: unicodedata.normalize(mode, text)
        else:
            self.print("No normalization will be used!")
            self.normalizer = lambda text, mode: text

    def _unicode_normalizer(self, text: str, mode: bool = None):
        mode = self.normalization_mode if mode is None else mode
        text = self.normalizer(text, mode)
        text = text.replace("\u200c", "").replace("\u200d", "")
        return text

    def ids_to_token(self, ids: list):
        """Convert ids to tokens
        Args:
            ids (list): token ids list
        Returns:
            list: tokens list
        """
        if not ids:
            raise ValueError("ids must be non-empty")

        if not isinstance(ids[0], list):
            token_list = [
                        self.vocab[idx] 
                        for idx in ids 
                        if self.vocab[idx] != self.blank_token and
                            self.vocab[idx] != self.bos_token and
                            self.vocab[idx] != self.eos_token
                        ]
                
            return token_list

        if isinstance(ids[0], list):
            return list(map(self.ids_to_token, ids))

    def ids_to_text(self, ids: list):
        """Convert token id list to a list of text
        Args:
            ids (list): Token id list
        Returns:
            list: A list of Text
        """

        if not ids:
            raise ValueError("ids must be non-empty")

        tokens = self.ids_to_token(ids)

        if not isinstance(tokens[0], list):
            return "".join(tokens)

        if isinstance(tokens[0], list):
            return list(map("".join, tokens))

    def _get_index(self, text, token):
        index = self.word2index.get(token)
        self.frequency_counter[token] += 1
        if index:
            return index

        self.out_of_vocabulary_info[token].add(text)
        return self.word2index[self.oov_token]

    def reset_out_of_vocabulary_info(self, keys=None):
        """Pops/resets out_of_vocabulary_info based on keys
        Args:
            keys (str|list, optional): a 1d list of strings or 'all'.
        """

        if isinstance(keys, list):
            for k in keys:
                self.out_of_vocabulary_info.pop(k, None)
            return
        if isinstance(keys, str):
            if keys.lower() == "all":
                self.out_of_vocabulary_info = defaultdict(set)
            return
        self.print("please,read function docs")

    def reset_frequency_counter(self, keys=None):
        """Pops/resets counter state based on keys
        Args:
            keys (str|list, optional): a 1d list of strings or 'all'.
        """

        if isinstance(keys, list):
            for k in keys:
                self.frequency_counter.pop(k, None)
            return
        if isinstance(keys, str):
            if keys.lower() == "all":
                self.frequency_counter = defaultdict(int)
            return
        self.print("please,read function docs")

    @property
    def unused_tokens(self):
        """returns a list token with zero frequency"""
        return sorted(list(set(self.vocab).difference(self.frequency_counter.keys())))

    def most_frequent_tokens(self, topk: int = None):
        """returns a dict with frequency of out of vocobulary tokens"""
        topk = topk if topk else len(self.frequency_counter)
        return {
            key: self.frequency_counter[key]
            for key in sorted(
                self.frequency_counter, key=self.frequency_counter.get, reverse=True
            )[:topk]
        }

    @property
    def out_of_vocobulary_tokens(self):
        """returns a list out of vocobulary tokens"""
        out_of_vocobulary_tokens = list(self.out_of_vocabulary_info.keys())
        return out_of_vocobulary_tokens

    @property
    def out_of_vocobulary_frequency(self):
        """returns a dictionary containing frequency of out of vocobulary tokens"""
        return {
            k: self.frequency_counter.get(k, -1) for k in self.out_of_vocobulary_tokens
        }

    def save_out_of_vocobulary_info(self, path_file_name="out_of_vocabulary_info"):
        """Save out of vocab tokens list as json
        Args:
            path_file_name (str, optional): Saving path. Defaults to 'out_of_vocabulary_info'
        """
        self._save_as_json(dict(self.out_of_vocabulary_info), path_file_name)

    def save_vocab(self, path_file_name="vocobulary"):
        """Save vocab tokens list as json
        Args:
            path_file_name (str, optional): Saving path. Defaults to 'out_of_vocabulary_info'
        """

        self._save_as_json(self.vocab, path_file_name)

    def load(self, path: str):
        """Load tokeinizer from disk. Currently supports pickle format only
        Args:
            path (str): Saved tokenizer path
        """
        with open(path, "rb") as file:
            data = pickle.load(file)

        missing_attr = [k for k in data.keys() if k not in self.__dict__]
        self.__dict__.update(data)

        if missing_attr:
            self.print(f"These attributes are missing: {missing_attr}")
        else:
            self.print("Tokenizer loaded successfully.")

    def save(self, path="tokenizer_object", format: str = "pickle"):
        """Save tokeinzer state to disk

        Args:
            path (str, optional): Saving path. Defaults to 'tokenizer_object'.
            format (str, optional): [pickle]. Defaults to 'pickle'.
        """

        if format == "pickle":
            self_ = {
                key: value
                for key, value in self.__dict__.items()
                if not key.startswith("__") and not callable(key)
            }
            path = path[:-4] if path.endswith(".pkl") else path
            _time = self._get_time_as_string()

            path = f"{path}_{_time}.pkl"
            with open(path, "wb") as file:
                pickle.dump(self_, file)

            self.print(f"tokenizer object has been save to {path}")

    def _save_as_json(self, data, file_name, cls=SetEncoder, encoding="utf-8"):
        _time = self._get_time_as_string()
        path_file_name = f"{file_name}_{_time}.json"

        with open(path_file_name, "w", encoding=encoding) as file:
            json.dump(data, file, ensure_ascii=False, cls=cls)

            self.print(f"File saved at {path_file_name}.")

    def _get_time_as_string(self):
        return datetime.now().strftime("%Y%m%d_%H.%M.%S")
