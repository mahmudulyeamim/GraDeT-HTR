from GraDT_HTR.config import DTrOCRConfig
from GraDT_HTR.model import DTrOCRLMHeadModel
from GraDT_HTR.processor import DTrOCRProcessor
from GraDT_HTR.utils import load_final_model

from PIL import Image
import torch

import time

import os
from tqdm import tqdm

ENG2BEN_MAP = str.maketrans({
    "0":"০","1":"১","2":"২","3":"৩","4":"৪",
    "5":"৫","6":"৬","7":"৭","8":"৮","9":"৯"
})

def load_extraction_model(
        root_path='',
        weights="BnDTrOCR/model_weights/model3_wo_pretrain_bntok_word.pth",
        device="cpu",
):
    config = DTrOCRConfig(
        bn_vocab_file=os.path.join(root_path, 'tokenization/bn_grapheme_1296_from_bengali.ai.buet.txt')
    )
    model = DTrOCRLMHeadModel(config)

    processor = DTrOCRProcessor(config)

    model = load_final_model(model, weights)

    model.eval()

    model.to(device)

    if device != "cpu":
        print(f"Moved the extraction model to: {next(model.parameters()).device}")

    return model, processor


# ─── sort helper ───────────────────────────────────────────────────────────────
def sort_underscore_numbers(keys):
    """
    Sorts a list of strings like '1_1_1_10', '1_1_1_2', etc. by their numeric components.
    Works for any number of underscore-separated parts.
    """
    return sorted(keys, key=lambda s: tuple(int(p) for p in s.split('_')))

# ─── single-word inference ─────────────────────────────────────────────────────
def extract_word_text(path_to_image, model, processor, device="cpu"):
    """
    Runs model+processor on one word image and returns the decoded text.
    """
    image = Image.open(path_to_image).convert("RGB")
    inputs = processor(
        images=image,
        texts=processor.tokeniser.bos_token,
        return_tensors="pt"
    )

    if inputs.pixel_values is not None:
        inputs.pixel_values = inputs.pixel_values.to(device)
    if inputs.input_ids is not None:
        inputs.input_ids = inputs.input_ids.to(device)
    if inputs.attention_mask is not None:
        inputs.attention_mask = inputs.attention_mask.to(device)
    
    model_output = model.generate(
        inputs=inputs,
        processor=processor,
        num_beams=3,
        use_cache=True
    )
    # replace the trocr whitespace token
    return processor.tokeniser.decode(model_output[0]).replace("▁", "").strip().translate(ENG2BEN_MAP)

# ─── process one line directory ────────────────────────────────────────────────
def process_line_dir(line_dir, model, processor, device="cpu"):
    """
    Given a directory containing word images (e.g. 1_1_1/),
    sorts them numerically by basename, runs inference on each,
    and concatenates into a single string.
    """
    # list only image files
    files = [f for f in os.listdir(line_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # map basename->filename
    base2file = {os.path.splitext(f)[0]: f for f in files}
    sorted_bases = sort_underscore_numbers(base2file.keys())
    
    words = []
    for base in sorted_bases:
        img_path = os.path.join(line_dir, base2file[base])
        words.append(extract_word_text(img_path, model, processor, device=device))
    
    return " ".join(words)

# ─── process entire page ───────────────────────────────────────────────────────
def process_page_dir(page_dir, model, processor, device="cpu"):
    """
    Given a page directory (e.g. '1_1'), finds all subdirectories,
    sorts them as lines, processes each line, then joins with newlines.
    """
    # only keep dirs
    subdirs = [d for d in os.listdir(page_dir)
               if os.path.isdir(os.path.join(page_dir, d))]
    sorted_lines = sort_underscore_numbers(subdirs)
    
    lines = []
    for line in sorted_lines:
        line_path = os.path.join(page_dir, line)
        lines.append(process_line_dir(line_path, model, processor, device))
    
    return "\n".join(lines)


def extract_full_page(page_dir, model, processor, device="cpu"):
    full_text = process_page_dir(page_dir, model, processor, device=device)
    return full_text


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = load_extraction_model(device=device)

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    
    print("Starting page processing.")

    PAGE_DIR = "BN_DRISHTI/content/final_word_segmentation"

    full_text = extract_full_page(PAGE_DIR, model, processor, device=device)

    print(full_text)
    end_time = time.time()

    print(f"Total time taken: {end_time - start_time}")

    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory usage: {peak_bytes/1024**3:.2f} GB")

