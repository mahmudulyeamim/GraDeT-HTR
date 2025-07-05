import marisa_trie


class MarisaTokenizer:
    def __init__(self, vocabulary, separator=None):
        self.trie = marisa_trie.Trie(
            [v[0] if isinstance(v, tuple) else v for v in vocabulary]
        )
        self.max_miss = 1

    def tokenize(self, text):
        words = []

        idx = 0
        while idx < len(text):
            length = 1
            continue_loop = False
            missed_chars = 0
            len_of_last_found_token = 1
            token_found = False

            while True:
                token = text[idx : idx + length]
                if self.trie.has_keys_with_prefix(token):
                    len_of_last_found_token = length
                    token_found, continue_loop = True, True
                else:
                    missed_chars += 1
                    continue_loop = missed_chars < self.max_miss

                length += 1

                if idx + length > len(text):
                    continue_loop = False

                if not continue_loop:
                    break

            if token_found:
                token = text[idx : idx + len_of_last_found_token]
                if token:
                    words.append(token.strip())
                    idx += len_of_last_found_token - 1

            idx += 1

        return words
