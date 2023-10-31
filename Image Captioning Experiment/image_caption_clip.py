# image_caption_clip.py

import os
import json
import io
import numpy as np
import torch
import faiss
import glob
import requests
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from CLIP import clip
from PIL import Image
from torchvision.utils import make_grid
import model
import retrofit
import fitz
import time
import psutil
import matplotlib.pyplot as plt  # Import the matplotlib library

# Define the dotdict class
class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Helper functions

# def load_clip_model(device):
#     clip_model = 'ViT-B/16'
#     return clip.load(clip_model, jit=False)[1].to(device)


def load_clip_model(device):
    clip_model = 'ViT-B/16'
    model, preprocess = clip.load(clip_model, jit=False)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def load_retrofit_model(device):
    config_path = './checkpoints/12xdqrwd-config'
    retrofit_ckpt = './checkpoints/12xdqrwd.ckpt'
    config = dotdict(torch.load(config_path))
    config.task = 'txt2txt'
    config.adapter = retrofit_ckpt
    net = retrofit.load_params(config).to(device)
    return net

def load_indices(index_dirs):
    indices = []
    indices_data = []
    for index_dir in index_dirs:
        fname = os.path.join(index_dir, 'args.txt')
        with open(fname, 'r') as f:
            index_args = dotdict(json.load(f))

        entries = []
        fname = os.path.join(index_dir, 'entries.txt')
        with open(fname, 'r') as f:
            entries.extend([line.strip() for line in f])

        indices_data.append(entries)
        indices.append(faiss.read_index(glob.glob(f"{index_dir}/*.index")[0]))

    return indices, indices_data

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')
def load_image(img, preprocess):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = Image.open(fetch(img))
    return img, preprocess(img).unsqueeze(0).to(device)

def clip_rescoring(args, net, candidates, x):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    textemb = net.perceiver.encode_text(
        clip.tokenize(candidates).to(args.device)).float()
    textemb /= textemb.norm(dim=-1, keepdim=True)
    similarity = (100.0 * x @ textemb.T).softmax(dim=-1)
    _, indices = similarity[0].topk(args.num_return_sequences)
    return [candidates[idx] for idx in indices[0]]

def caption_image(net, args, image_path, preprocess, context):
    captions = []
    img, mat = load_image(image_path, preprocess)
    device = args.device
    table, x = net.build_table(mat.half(),
                               net.perceiver,
                               ctx=context,
                               indices=net.indices,
                               indices_data=net.indices_data,
                               knn=args.knn,
                               tokenize=clip.tokenize,
                               device=device,
                               is_image=True,
                               return_images=True)

    table = net.tokenizer.encode(table[0], return_tensors='pt').to(device)
    table = table.squeeze()[:-1].unsqueeze(0)
    out = net.model.generate(table,
                             max_length=args.maxlen,
                             do_sample=args.do_sample,
                             num_beams=args.num_beams,
                             temperature=args.temperature,
                             top_p=args.top_p,
                             num_return_sequences=args.num_return_sequences)
    candidates = []
    for seq in out:
        decoded = net.tokenizer.decode(seq, skip_special_tokens=True)
        decoded = decoded.split('|||')[1:][0].strip()
        candidates.append(decoded)
    captions = clip_rescoring(args, net, candidates, x[None, :])
    cap = ""
    for c in captions[:args.display]:
        print(c)
        cap = cap + c
    return cap
# Define a function to measure memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    return memory_usage

def extract_images_from_pdf(pdf_url, output_format="png", min_width=1, min_height=1, output_dir="extracted_images", args=None, net=None, preprocess=None):
    response = requests.get(pdf_url)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    if response.status_code == 200:
        pdf_content = response.content
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

        extracted_data = {}
        image_indices = []
        times = []
        memory_usages = []
        for page_index in range(len(pdf_document)):
            page = pdf_document[page_index]
            image_list = page.get_images(full=True)

            for image_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))

                if image.width >= min_width and image.height >= min_height:
                    print(f"Image {image_index} on page {page_index + 1} ")
                    image_path = os.path.join(output_dir, f"image{page_index + 1}_{image_index}.{output_format}")
                    image.save(open(image_path, "wb"), format=output_format.upper())
                    display(image)

                    # caption = caption_image(net, args, image_path, preprocess, context=['Educated'])
                    start_time_caption = time.time()  # Record start time for caption generation

                    caption = caption_image(net, args, image_path, preprocess, context=['Educated'])

                    end_time_caption = time.time()  # Record end time for caption generation

                    memory_usage = get_memory_usage()  # Get memory usage after caption generation

                    # Calculate time taken for caption generation
                    time_taken_caption = end_time_caption - start_time_caption


                    extracted_data[f"Image {image_index} on page {page_index + 1} "] = caption

                    image_indices.append(f"Image {image_index} on page {page_index + 1}")
                    times.append(time_taken_caption)
                    memory_usages.append(memory_usage)
                    print(f"Time taken for caption generation: {time_taken_caption:.2f} seconds")
                    print(f"Memory usage after caption generation: {memory_usage:.2f} MB")
                else:
                    print(f"[-] Skipping image {image_index} on page {page_index} due to its small size.")
        # Create a plot with time, memory usage, and image index
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(image_indices, times, marker='o', linestyle='-')
        plt.title('Time taken for caption generation')
        plt.xlabel('Image Index')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=90)

        plt.subplot(2, 1, 2)
        plt.plot(image_indices, memory_usages, marker='o', linestyle='-')
        plt.title('Memory usage after caption generation')
        plt.xlabel('Image Index')
        plt.ylabel('Memory (MB)')
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken: {time_taken:.2f} seconds")
        return extracted_data

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = dotdict(
        knn=3,
        maxlen=72,
        num_return_sequences=90,
        num_beams=1,
        temperature=0.8,
        top_p=0.9,
        display=5,
        do_sample=True,
        device=device
    )

    indices, indices_data = load_indices(['./unigrams', './bigrams', './artstyles', './emotions'])
    preprocess = clip.load('ViT-B/16', jit=False)[1]
    clip_model = load_clip_model(device)
    retrofit_model = load_retrofit_model(device)
    retrofit_model.indices = indices
    retrofit_model.indices_data = indices_data

    pdf_url = "https://www.example.com/sample.pdf"
    output_format = "png"
    min_width = 1
    min_height = 1
    output_dir = "extracted_images_ofa"

    extracted_data = extract_images_from_pdf(pdf_url, output_format, min_width, min_height, output_dir, args, retrofit_model, preprocess)

    # Use the extracted_data dictionary as needed
