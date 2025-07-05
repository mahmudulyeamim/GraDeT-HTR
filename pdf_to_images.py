from pdf2image import convert_from_path
import os


def pdf_to_images(pdf_path, output_dir, number):
    """
    Converts a PDF into distinct images using pdf2image and saves them.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the output images.
        image_format (str): Format to save images (e.g., 'JPEG', 'PNG').

    Returns:
        None
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to images
    pages = convert_from_path(pdf_path)
    
    # Save each page as an image file
    for i, page in enumerate(pages):
        output_file = os.path.join(output_dir, f"{number}_{i + 1}.jpg")
        page.save(output_file, 'JPEG')
        print(f"Saved: {output_file}")
        
    os.remove(pdf_path)
    print(f"Deleted: {pdf_path}")


if __name__ == "__main__":
    # Path to the input PDF file
    pdf_path = f"input_pages/1.pdf"

    # Directory to save output images
    output_dir = f"input_pages"

    # Call the function
    pdf_to_images(pdf_path, output_dir, 1)
