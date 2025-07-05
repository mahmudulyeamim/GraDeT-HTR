import sys
sys.path.append('../')

import torch

from BnGraphemizer.trie_tokenizer import TrieTokenizer
from BnGraphemizer.base import GraphemeTokenizer

class BnGraphemizerProcessor:
    def __init__(
        self, 
        grapheme_file, 
        model_max_length=32, 
        normalize_unicode=True, 
        normalization_mode='NFKC', 
        normalizer="buetNormalizer",
        blank_token: str = "_",
        bos_token: str = "_",
        eos_token: str = "_",
        add_bos_token=False,
        add_eos_token=False):
        
        self.grapheme_file = grapheme_file
        self.model_max_length = model_max_length
        self.normalize_unicode = normalize_unicode
        self.normalization_mode = normalization_mode
        self.normalizer = normalizer
        self.blank_token = blank_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.list_of_graphemes = self._load_graphemes()
        self.bn_graphmemizer = self._initialize_graphemizer()
        
        self.pad_token_id = self.bn_graphmemizer.pad_token_id
        self.bos_token_id = self.bn_graphmemizer.bos_token_id
        self.eos_token_id = self.bn_graphmemizer.eos_token_id

    def _load_graphemes(self):
        """Load the graphemes from the file."""
        with open(self.grapheme_file, 'r', encoding='utf-8') as f:
            graphemes = sorted(
                list(set(
                    [line.strip() for line in f.readlines() if line.strip()]
                ))
            )
        return graphemes

    def _initialize_graphemizer(self):
        """Initialize the graphemizer with the loaded graphemes."""
        graphemizer = GraphemeTokenizer(
            tokenizer_class=TrieTokenizer,
            max_len=self.model_max_length,
            normalize_unicode=self.normalize_unicode,
            normalization_mode=self.normalization_mode,
            normalizer=self.normalizer,
            printer=print,
            blank_token=self.blank_token,
            bos_token = self.bos_token,
            eos_token = self.eos_token,
            add_bos_token=self.add_eos_token,
            add_eos_token=self.add_eos_token
        )
        graphemizer.add_tokens(self.list_of_graphemes, reset_oov=True)
        return graphemizer

    def __call__(self, texts, padding=False):
        """Tokenize a list of Bengali texts."""
        bng_text_inputs = self.bn_graphmemizer.tokenize(texts, padding=padding)
        
        # print(bng_text_inputs)
        
        bng_inputs = self._get_tokenized_inputs(bng_text_inputs)
        # print(bng_inputs)

        bng_input_ids = torch.Tensor(bng_inputs['input_ids']).long()
        bng_attention_mask = torch.Tensor(bng_inputs['attention_mask']).long()
        
        if bng_input_ids.ndim == 1:
            bng_input_ids = bng_input_ids.unsqueeze(0)
        if bng_attention_mask.ndim == 1:
            bng_attention_mask = bng_attention_mask.unsqueeze(0)

        return {
            'input_ids': bng_input_ids,
            'attention_mask': bng_attention_mask
        }
        
    def _get_tokenized_inputs(self, inputs):
        """Split the tokenized inputs into input_ids and attention_mask"""
        if not isinstance(inputs, list):
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            } 
            
        input_ids = []
        attention_mask = []
        for input in inputs:
            if isinstance(input, list):
                input = self._get_tokenized_inputs(input)
            input_ids.append(input['input_ids'])
            attention_mask.append(input['attention_mask'])
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        } 

        
    def decode(self, input_ids):
        """Decode input IDs back to the original text."""
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor.")

        if input_ids.ndim == 0:
            input_ids = input_ids.unsqueeze(0)

        if input_ids.ndim == 1:
            token_list = self.bn_graphmemizer.ids_to_token(input_ids.tolist())
            decoded_text = ''.join(token_list)

            return decoded_text
        elif input_ids.ndim >= 2:
            return [
                self.decode(input_ids[i]) for i in range(input_ids.shape[0])
            ]
        else:
            raise ValueError("Unsupported input tensor dimensions.")
    

if __name__ == "__main__":
    processor = BnGraphemizerProcessor(grapheme_file='bn_grapheme_1296_from_bengali.ai.buet.txt', add_bos_token=True, add_eos_token=True)

    bng_texts = [["শুভ অপরাহ্ন", "পরে দেখা হবে", "শুভ জন্মদিন", "অভিনন্দন"],
                 ["শুভ অপরাহ্ন", "পরে দেখা হবে", "শুভ জন্মদিন", "অভিনন্দন"]]
    
    tokenized_outputs = processor(bng_texts, padding=True)

    print(tokenized_outputs['input_ids'])
    print(tokenized_outputs['input_ids'].shape)
    print(tokenized_outputs['input_ids'].dtype)

    print(tokenized_outputs['attention_mask'])
    print(tokenized_outputs['attention_mask'].shape)
    print(tokenized_outputs['attention_mask'].dtype)

    decoded_outputs = processor.decode(tokenized_outputs['input_ids'])
    print(decoded_outputs)
