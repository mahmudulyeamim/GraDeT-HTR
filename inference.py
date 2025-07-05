from segment_single_page import load_segmentation_models, run_segmentation_model
from extract_single_page import load_extraction_model, extract_full_page, sort_underscore_numbers
from pdf_to_images import pdf_to_images
import time
import torch
import os
import shutil


def clean_workspace(directory):
    if os.path.isdir(directory):
        # Walk through all subdirectories and files in the directory
        for root, dirs, files in os.walk(directory, topdown=False):
            # Delete all files in the current directory
            for file in files:
                os.remove(os.path.join(root, file))
            # Delete all subdirectories
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))
        print(f"All files and subdirectories in '{directory}' have been deleted.")
    else:
        print(f"The provided path '{directory}' is not a valid directory.")
        
        
def segmentation(
    input_dir,
    line_seg_model_config,
    word_seg_model_config,
    segmented_page_dir,
    root_path='',
    pdf_flag=False,
):
    if pdf_flag:
        pdf_to_images(
            pdf_path=input_dir + '/' + '1.pdf',
            output_dir=input_dir,
            number=1
        )
    
    for page in os.listdir(input_dir):
        image_label, _ = os.path.splitext(page) # ('1_1', '.jpg')
        image_path = input_dir + '/' + page # input_images/1_1.jpg
        final_word_segmentation_dir = segmented_page_dir + '/' + image_label + '/' # output_segmentations/1_1/
        run_segmentation_model(
            image_path=image_path,
            image_label=image_label,
            line_model_config=line_seg_model_config,
            word_model_config=word_seg_model_config,
            final_word_segmentation=final_word_segmentation_dir,
            root_path=root_path
        )


def extraction(
    model,
    processor,
    segmented_page_dir,
    output_dir,
    device="cpu",
):
    for dir in sort_underscore_numbers(os.listdir(segmented_page_dir)):
        page_dir = segmented_page_dir + '/' + dir # outputs_segmentations/1_1
        
        full_text = extract_full_page(page_dir, model, processor, device=device)

        txt_path = os.path.join(output_dir, f"{dir}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print("Output saved to ", txt_path)


if __name__ == "__main__":
    '''
    in case of image(s) input:
        place the input images in the input_dir
            the name of the images should be 1_1, 1_2, ...
    in case of pdf input:
        place the input pdf in the input_dir
            the name of the pdf should be 1.pdf
    '''
    
    # paths
    root_path = '' # root path which is added as prefix to all the relative paths
    input_dir  = os.path.join(root_path, 'input_pages') # the inputs are expected to be in this directory
    segmented_page_dir = os.path.join(root_path, "output_segmentations") # segmented words will be saved here
    output_dir = os.path.join(root_path, 'output_texts') # final text detection results will be saved here
    
    # flags
    pdf_flag = False # turn to true when the input is a pdf.
    clean_input_directory = False # turn to true if you wanna clean the input directory after extracting text
    
    # code starts
    clean_workspace(segmented_page_dir)
    clean_workspace(output_dir)

    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --------------------------------- loading models starts --------------------------------------
    print("Loading segmentation models...")
    line_seg_model_config, word_seg_model_config = load_segmentation_models(
        line_weights=os.path.join(root_path, 'BN_DRISHTI/model_weights/line_model_best.pt'),
        word_weights=os.path.join(root_path, 'BN_DRISHTI/model_weights/word_model_best.pt'),
        half=False,
        device=device
    )
    print("Segmenetation models loaded successfully")
    print("Loading text extraction model...")
    model, processor = load_extraction_model(
        root_path=root_path,
        weights=os.path.join(root_path, "GraDT_HTR/model_weights/model3_wo_pretrain_bntok_word.pth"),
        device=device
    )
    print("Text extraction model loaded successfully")
    # --------------------------------- loading models ends --------------------------------------


    # --------------------------------- inferece starts --------------------------------------
    t0 = time.time()

    print("Starting page segmentation....")
    
    segmentation(
        input_dir,
        line_seg_model_config,
        word_seg_model_config,
        segmented_page_dir,
        root_path,
        pdf_flag
    )

    t1 = time.time()
    
    print("Starting text extraction....")
    
    extraction(
        model,
        processor,
        segmented_page_dir,
        output_dir,
        device=device
    )

    t2 = time.time()
    # --------------------------------- inferece ends --------------------------------------

    if clean_input_directory:
        clean_workspace(input_dir)

    print("Time to segment: ", t1 - t0)
    print("Time to extract: ", t2 - t1)

    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory usage: {peak_bytes/1024**3:.2f} GB")
