# -*- coding: utf-8 -*-
"""vit_gpt2_Function.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s0ovjHkRgfaXVAmdfhtTNZhQePTFGa66
"""

import os
import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
import cv2
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
import requests
import matplotlib.pyplot as plt
import re
import transformers
from google.colab.patches import cv2_imshow
import nltk
from nltk.translate.bleu_score import sentence_bleu
from google.colab.patches import cv2_imshow  # Import cv2_imshow for image display in Colab
import time
import psutil
# Download the 'punkt' resource
nltk.download('punkt')
import time

def extract_images_and_text_from_pdf_vit(self, min_width=1, min_height=1, output_dir="extracted_images"):
    start_time = time.time()
    # Load the image captioning model and processor
    model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to load the model: {time_taken:.2f} seconds")
    memory_usage_bytes = psutil.virtual_memory().used

    # Convert to megabytes (MB)
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # 1 MB = 1024 * 1024 bytes
    print(f"Memory Usage to load the model: {memory_usage_mb:.2f} MB")
    pdf_url=self.url
    # Retrieve the PDF content from the URL
    response = requests.get(pdf_url)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_time = time.time()

    if response.status_code == 200:
        pdf_content = response.content

        # Create a PyMuPDF document object from the PDF content
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

        # Define a regular expression pattern to match numbers
        number_pattern = r'\d+(\.\d+)%'
        cap={}
        memory_usage = []
        time_taken_list = []

        # Iterate over PDF pages and extract images
        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]
            image_list = page.get_images(full=True)
            if image_list:
                print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
            for image_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Load image as a NumPy array
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

                # Check if the image meets the minimum dimensions
                if image.shape[1] >= min_width and image.shape[0] >= min_height:
                    # Generate a caption for the image
                    pixel_values = image_processor(image, return_tensors="pt").pixel_values

                    generated_ids = model_raw.generate(pixel_values, max_new_tokens=30)
                    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    # Show the processed image
                    cv2_imshow(image)

                    # Perform OCR to extract text from the image
                    extracted_text = pytesseract.image_to_string(image)

                    # Save the processed image to the output directory
                    image_path = os.path.join(
                        output_dir, f"image{page_index + 1}_{image_index}.png")
                    cv2.imwrite(image_path, image)

                    # Print the extracted text, classification, and the generated caption
                    print(f"Image {image_index} on page {page_index + 1} ")
                    print("The generated caption on the image: ", generated_text)

                    if extracted_text.strip():
                      print(f"The image contains: {extracted_text}")
                    else :
                      extracted_text=" "
                    cap[f"Image {image_index} on page {page_index + 1} "]=generated_text +' ' + extracted_text
                    # Track time taken for each image processing
                    time_taken_list.append(time.time() - start_time)
                    # Track memory usage
                    memory_usage.append(psutil.virtual_memory().percent)
                    print(f"Memory Usage: {memory_usage[-1]:.2f}")
                    print(f"Time Taken for Caption Generation: {time_taken_list[-1]:.2f} seconds")

                else:
                    print(
                        f"[-] Skipping image {image_index} on page {page_index + 1} due to its small size.")
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")
        # Plot memory and time data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(memory_usage)
    plt.title('Memory Usage for Each Image ')
    plt.xlabel('Image Index')
    plt.ylabel('Memory Usage (MB)')

    plt.subplot(1, 2, 2)
    plt.plot(time_taken_list)
    plt.title('Time Taken for Each Image')
    plt.xlabel('Image Index')
    plt.ylabel('Time Taken (seconds)')

    plt.tight_layout()
    plt.show()


    return cap