class TrieTokenizer:
    def __init__(self, vocabulary, separator=None):
        self.trie = self._make_trie(
            [v[0] if isinstance(v, tuple) else v for v in vocabulary]
        )

    def _make_trie(self, words):
        root = dict()
        for word in words:
            if word:
                current_node = root.get(word[0], {"isTerminal": len(word) == 0})
                root[word[0]] = self._add_token(word[1:], current_node)
        return root

    def _add_token(self, token, current_node):
        if not token:
            current_node["isTerminal"] = True
            return current_node
        else:
            new_node = current_node.get(token[0], {"isTerminal": len(token) == 0})
            current_node[token[0]] = self._add_token(token[1:], new_node)
            return current_node

    def tokenize(self, text):
        tokenized_text = []
        while text:
            token, text = self._get_next_token(text, self.trie)
            tokenized_text.append(token)

        return tokenized_text

    def _get_next_token(self, text, trie):
        terminal_idx = 0
        current_trie = trie
        for idx, char in enumerate(text):
            if char not in current_trie:
                break
            else:
                current_trie = current_trie.get(char, {})
                if current_trie.get("isTerminal", False):
                    terminal_idx = idx

        return text[: terminal_idx + 1], text[terminal_idx + 1 :]
