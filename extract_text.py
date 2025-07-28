# %%
from pdf2image import convert_from_path
import os

def convert_pdf_to_images(pdf_path, output_folder="pdf_images", dpi=300):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF pages to images
    print(f"📄 Converting PDF: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=dpi)

    # Save each image as PNG
    for i, img in enumerate(images, start=1):
        image_path = os.path.join(output_folder, f"page_{i}.png")
        img.save(image_path, "PNG")
        print(f"✅ Saved: {image_path}")

    print(f"\n🎉 Done! {len(images)} pages saved to '{output_folder}'.")

# === USAGE ===
convert_pdf_to_images("/home/sovan/harsh/class12_phy_ch8/ch8class12phy.pdf", output_folder="/home/sovan/harsh/class12_phy_ch8/chap8_img_pages")

# %%
from pdf2image import convert_from_path
import openai
openai.api_key = "sk-proj-wzNvN7"

import base64

import re

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def extract_from_image(image_path):
    base64_image = encode_image(image_path)


    prompt = (
    "Extract the exact content from the page to create a faithful replica, suitable for semantic embedding. Follow these strict guidelines:\n\n"
    "1. Convert all mathematical expressions into plain natural language.\n"
    "   - Do NOT use LaTeX formatting (like \\( ... \\) or \\[ ... \\]).\n"
    "   - Instead, rewrite equations in simple English (e.g., write 'current equals charge divided by time' instead of 'I = Q/t').\n\n"
    "2. If there are diagrams or images:\n"
    "   - Write only a plain English caption describing what the figure shows.\n"
    "   - Do not speculate or add interpretation beyond what is clearly visible.\n\n"
    "3. Identify headings using the following rules:\n"
    "   - Enclose ALL CAPS headings in [square brackets], including headings that contain numbers or punctuation (e.g., '5.2 CYCLE').\n"
    "   - Enclose headings written in lowercase, title case, or sentence case in {curly braces}. These are usually subheadings.\n"
    "   - Only mark headings when they are visually distinct from body text, such as by font size, boldness, or spacing.\n\n"
    "4. Extract the page number and include it at the end of the content inside angle brackets. For example: <Page 37>\n\n"
    "5. Extract all tables from the provided textbook page image and convert their contents into clear, complete English sentences or simple lists. Follow these guidelines:\n"
    "Do not use any LaTeX or mathematical notation. Write out equations and numeric relationships in plain English.\n"
    "   - Carefully capture all information from the table, including headers and all cells, without omitting or summarizing any details.\n"
    "   - For each table, structure the information as full sentences or straightforward, linear lists that are easy to understand and ready for text embedding.\n"
    "   - Do not attempt to interpret, infer, or add any information beyond what is clearly present in the table.\n"
    "   - Exclude all OCR noise, page numbers, headers, footers, or unrelated text.\n"
    "   - If the table presents structured data (like categories or classifications), ensure the relationships are described explicitly in your output.\n"
    "   - Maintain the order and the grouping of information as in the original table.\n\n"
    "6. Ignore all OCR noise, headers, footers, or unrelated artifacts.\n\n"
    "7. Do not generate or infer any additional information. Include only what is explicitly present on the page."
)
    
    response = openai.chat.completions.create(
        model="gpt-4o",  # or "gpt-4-vision-preview"
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }}
                ]
            }
        ],
        max_tokens=2000,
    )
    
    return response.choices[0].message.content


# %%
import os

def convertImgToTxt(page):
    os.makedirs("chap7_txt_pages", exist_ok=True)  

    result = extract_from_image(f"/home/sovan/harsh/class12_phy_ch8/chap8_img_pages/{page}.png")
    with open(f"/home/sovan/harsh/class12_phy_ch8/chap8_txt_pages/{page}.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print(f"Output for {page} saved")

cnt = 0
folder_path = "/home/sovan/harsh/class12_phy_ch8/chap8_img_pages"
file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
for i in range(file_count):
        cnt+=1
        convertImgToTxt(f"page_{i+1}")
print(f'\n\n\nConverted {cnt} image files to text files\n')

# %%
import re
import pickle
import os
import shutil

HEADINGS_PICKLE = "headings.pkl"

# Source pickle file
src = 'headings.pkl'

# Destination copy file
dst = 'headings_copy.pkl'


def load_previous_headings():
    if os.path.exists(HEADINGS_PICKLE):
        with open(HEADINGS_PICKLE, "rb") as f:
            return pickle.load(f)
    return {"main_heading": "Untitled", "sub_heading": None}

def save_current_headings(main_heading, sub_heading):
    with open(HEADINGS_PICKLE, "wb") as f:
        pickle.dump({"main_heading": main_heading, "sub_heading": sub_heading}, f)



def extract_and_chunk(text, chapter_name, file_name):
    headings = load_previous_headings()
    current_main = headings["main_heading"]
    current_sub = headings["sub_heading"]

    old_main = current_main
    old_sub = current_sub

    # Extract document page number from filename
    doc_page_match = re.search(r'(\d+)', file_name)
    doc_page_num = doc_page_match.group(1) if doc_page_match else "Unknown"

    # Extract source page from angle brackets in text
    source_page_match = re.search(r"<(.*?)>", text)
    source_page = source_page_match.group(1).strip() if source_page_match else "Unknown"

    # Find headings
    main_matches = [(m.start(), m.group(1).strip(), 'main') for m in re.finditer(r"\[(.*?)\]", text)]
    sub_matches = [(m.start(), m.group(1).strip(), 'sub') for m in re.finditer(r"\{(.*?)\}", text)]
    all_markers = sorted(main_matches + sub_matches, key=lambda x: x[0])

    chunks = []
    pointer = 0

    for i, (pos, value, tag_type) in enumerate(all_markers):
        chunk_text = text[pointer:pos].strip()

        # Remove heading markers from chunk text
        chunk_text_clean = re.sub(r"^\[.*?\]$", "", chunk_text).strip()
        chunk_text_clean = re.sub(r"^\{.*?\}$", "", chunk_text_clean).strip()

        if chunk_text_clean and chunk_text_clean != current_main and chunk_text_clean != current_sub:
            chunks.append({
                "chapter": chapter_name,
                "title": current_main,
                "section": current_sub,
                "source_page": source_page,
                "doc_page": doc_page_num,
                "content": chunk_text
            })

        # Update heading
        if tag_type == "main":
            current_main = value
            current_sub = None
        else:
            current_sub = value

        pointer = pos

    # Final chunk
    final_text = text[pointer:].strip()
    final_text_clean = re.sub(r"^\[.*?\]$", "", final_text).strip()
    final_text_clean = re.sub(r"^\{.*?\}$", "", final_text_clean).strip()

    if final_text_clean and final_text_clean != current_main and final_text_clean != current_sub:
        chunks.append({
            "chapter": chapter_name,
            "title": current_main,
            "section": current_sub,
            "source_page": source_page,
            "doc_page": doc_page_num,
            "content": final_text
        })

    # Save updated headings
    banned_words = ["activity", "figure"]
    if current_sub and any(bad in current_sub.lower() for bad in banned_words):
        save_current_headings(current_main, old_sub)
    else:
        save_current_headings(current_main, current_sub)

    return chunks

# %%
def createChunks(num):
    textFile = f"/home/sovan/harsh/class12_phy_ch8/chap8_txt_pages/page_{num}.txt"
    chapterName = "Electromagnetic Waves"
    fileName = f"page_{num}.txt"

    with open(textFile, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = extract_and_chunk(text, chapterName, fileName)

    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1} ---")
        print(f"Chapter     : {chunk['chapter']}")
        print(f"Title       : {chunk['title']}")
        print(f"Section     : {chunk['section']}")
        print(f"Source Page : {chunk['source_page']}")
        print(f"Doc Page    : {chunk['doc_page']}")
        print("Content     :")
        print(chunk['content'])
    
    with open(f"/home/sovan/harsh/class12_phy_ch8/chap8_pagewise_chunk/chunks_page_{num}.pkl", "wb") as f:
        pickle.dump(chunks, f)


cnt = 0
folder_path = "/home/sovan/harsh/class12_phy_ch8/chap8_txt_pages"
file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
for i in range(file_count):
        cnt+=1
        createChunks(i+1)

print(f'\n\nChunked {cnt} text files\n')

# %%
all_chunks = []
folder_path = "//home/sovan/harsh/class12_phy_ch8/chap8_pagewise_chunk"
file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

for i in range(file_count):
        pkfile = f"chunks_page_{i+1}.pkl"
        file = os.path.join(folder_path, pkfile)
        if os.path.isfile(file):
                with open(file, "rb") as f1:
                        chunk = pickle.load(f1)
                        all_chunks.extend(chunk)
                        
with open("all_chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print(f"Combined {i+1} chunked pages into one chunk ")

# %%
import pickle

chunk_file = "/home/sovan/harsh/all_chunks.pkl"

# Load the pickle and print the length of the list
with open(chunk_file, "rb") as f:
    all_chunks = pickle.load(f)

print(f"Total number of chunks stored: {len(all_chunks)}")

for i, chunk in enumerate(all_chunks):
    print(f"\n### Chunk {i + 1} ###")
    print(f"Chapter     : {chunk['chapter']}")
    print(f"Title       : {chunk['title']}")
    print(f"Section     : {chunk['section']}")
    print(f"Source Page : {chunk['source_page']}")
    print(f"Doc Page    : {chunk['doc_page']}")
    print("Content     :")
    print(chunk['content'])

# %%



