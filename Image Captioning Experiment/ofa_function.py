# -*- coding: utf-8 -*-
"""OFA_Function.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s0ovjHkRgfaXVAmdfhtTNZhQePTFGa66
"""

from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator

from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

import os
import requests
import fitz  # PyMuPDF
import io
import torch
import re
import time
import psutil
import matplotlib.pyplot as plt

def extract_images_from_pdf_ofa(self, output_format="png", min_width=1, min_height=1, output_dir="extracted_images"):
    
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    ckpt_dir='./OFA-huge'
    tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
    txt = " what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids

    # Record the start time
    start_time = time.time()
    model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
    # Record the end time
    end_time = time.time()

    # Calculate the time taken in seconds
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")
    memory_usage_bytes = psutil.virtual_memory().used
    # Convert to megabytes (MB)
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # 1 MB = 1024 * 1024 bytes
    print(f"Memory Usage: {memory_usage_mb:.2f} MB")
    
    generator = sequence_generator.SequenceGenerator(
    tokenizer=tokenizer,
    beam_size=5,
    max_len_b=16,
    min_len=0,
    no_repeat_ngram_size=3,
    )
    pdf_url=self.url
    # Retrieve the PDF content from the URL
    response = requests.get(pdf_url)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    if response.status_code == 200:
        pdf_content = response.content

        # Create a PyMuPDF document object from the PDF content
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        time_data = []
        memory_data = []
        # List to store extracted images
        extracted_images = []
        cap={}
        # Iterate over PDF pages
        for page_index in range(len(pdf_document)):
            # Get the page itself
            page = pdf_document[page_index]
            # Get image list
            image_list = page.get_images(full=True)
            # Print the number of images found on this page
            if image_list:
                print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
            # Iterate over the images on the page
            for image_index, img in enumerate(image_list, start=1):
                # Get the XREF of the image
                xref = img[0]
                # Extract the image bytes
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                # Get the image extension
                image_ext = base_image["ext"]
                # Load it to PIL
                image = Image.open(io.BytesIO(image_bytes))
                data = {}
                
               # Check if the image meets the minimum dimensions and save it
                if image.width >= min_width and image.height >= min_height:
                    image_start_time = time.time()
                    # Perform OCR to extract text from the image
                    patch_img = patch_resize_transform(image).unsqueeze(0)

                    data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
                    gen_output = generator.generate([model], data)
                    gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]
                    display(image)
                    # Perform OCR to extract text from the image
                    extracted_text = pytesseract.image_to_string(image)
                    image_end_time = time.time()
                    image_time_taken = image_end_time - image_start_time

                    # Define a regular expression pattern to match numbers
                    number_pattern = r'\d+(\.\d+)%'

                    # Use re.sub to replace numbers with an empty string
                    text_without_numbers = re.sub(number_pattern, '', extracted_text)
                    print(f"Image {image_index} on page {page_index + 1} ")
                    print("The generated cation on the image: ",tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())
                    
                    if extracted_text.strip():
                      print(f"The image contains: {extracted_text}")
                    else :
                      extracted_text=" "
                    cap[f"Image {image_index} on page {page_index + 1} "] = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip() +' ' + extracted_text
                    time_data.append(image_time_taken)  
                     # Measure memory usage and add it to the list using psutil
                    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # in MB
                    memory_data.append(memory_usage)
                    print(f"Time taken for generated captions {image_index}: {image_time_taken:.2f} seconds")
                    print(f"Memory usage for generated captions {image_index}: {memory_usage:.2f} MB")
                    
                    # Save image to the output directory
                    image_path = os.path.join(output_dir, f"image{page_index + 1}_{image_index}.{output_format}")
                    image.save(open(image_path, "wb"), format=output_format.upper())
                    # Add the image, its path, and extracted text to the list
                    extracted_images.append({"path": image_path, "image": image})
                    # Use 'extracted_text' as the caption for the image
                    # print(f"Caption for {image_path}: {extracted_text}")
                else:
                    print(f"[-] Skipping image {image_index} on page {page_index} due to its small size.")

        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken: {time_taken:.2f} seconds")
        # After processing all images, create a plot for time and memory data
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(time_data, marker='o')
        plt.title('Time Usage for Each Image')
        plt.xlabel('Image Index')
        plt.ylabel('Time (seconds)')

        plt.subplot(2, 1, 2)
        plt.plot(memory_data, marker='o')
        plt.title('Memory Usage for Each Image')
        plt.xlabel('Image Index')
        plt.ylabel('Memory (MB)')

        plt.tight_layout()
        plt.show()

    # else:
    #     print("Failed to retrieve PDF content from the URL.")

    return cap