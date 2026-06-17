import numpy as np
import cv2
import os
import gc
import io
import re
import random
import string
import pybase64
from typing import List,Tuple,Union, Optional, Dict, Any

def random_alphanumeric(length: int=6) -> str:
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))
    
def is_image_empty(image: Optional[np.ndarray]) -> bool:
    return image is None or (hasattr(image, 'size') and image.size == 0) or (isinstance(image, np.ndarray) and np.all(image == 0))

def im2b64(img: np.ndarray) -> str:
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    b64 = pybase64.b64encode(img_bytes).decode('utf-8')
    return b64

def display_similarity_results(similarity_array: np.ndarray, file_array: List[str], threshold: float) -> None:
    if len(similarity_array) != len(file_array):
        print("Error: Similarity array and file array must have the same length.")
        return
    for i, similarity in enumerate(similarity_array):
        file_name = file_array[i]
        if similarity >= threshold:
            print(f"Match: {file_name} - Similarity: {similarity:.4f}")
        else:
            print(f"No Match: {file_name} - Similarity: {similarity:.4f}")
 
def get_sim(emb1,emb2):
  dot_product = np.dot(emb1,emb2)
  norm_query = np.linalg.norm(emb2)
  norms = np.linalg.norm(emb1, axis=0)
  similarity = dot_product / (norm_query * norms + 1e-8)  # Avoid division by zero
  print("similarity",similarity)

def get_sim_array(embedding, embeddings_array):
    embedding = embedding / np.linalg.norm(embedding)
    embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    similarities = np.dot(embeddings_array, embedding)
    return similarities
  
#def get_emb_fromcrop(imPath):  
#  with open(imPath, "rb") as image2string:
#      decoded_bytes = image2string.read()
#  img_np = np.frombuffer(decoded_bytes, dtype=np.uint8)
#  cropped_face = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#  embed = detector.get_embedding(cropped_face)
#  return embed

def binary_quantize(embedding, threshold=0):
    embedding_np = np.array(embedding)  # Ensure it's a NumPy array
    binary_embedding = (embedding_np > threshold).astype(int)
    return binary_embedding

def hamming_distance(binary_embedding1, binary_embedding2):
    if len(binary_embedding1) != len(binary_embedding2):
        raise ValueError("Binary embeddings must have the same length.")

    return np.sum(binary_embedding1 != binary_embedding2)

def clear_vectors(vectors_variable):
    """Clears a loaded vectors variable and releases memory."""

    if vectors_variable is not None:
        try:
            del vectors_variable # Delete the variable reference
            gc.collect() # Force garbage collection
            print("Vectors variable cleared.")
        except NameError:
            print("Variable does not exist, or has already been cleared.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("Vectors variable was already None.")

def is_file_path(s):
    """Check if a string is a file path (even if it doesn't exist) and not a Base64 string."""
    # Check if it contains a path separator or file extension
    if os.path.sep in s or re.search(r"\.\w{1,5}$", s):  
        return True
    
    # Base64 strings only contain A-Z, a-z, 0-9, +, /, and may end with =
    base64_pattern = re.fullmatch(r"[A-Za-z0-9+/=]+", s)
    return base64_pattern is None  # If it doesn't match Base64, it's likely a file path

def load_image(image_input: Union[str, bytes]) -> Optional[np.ndarray]:
    if isinstance(image_input, str) and os.path.exists(image_input):
        return cv2.imread(image_input)
    try:
        print(is_file_path(image_input))
        if is_file_path(image_input):
          print(f"File : {image_input} not found")
          return None
        decoded_bytes = pybase64.b64decode(image_input)
#        with open("./testIm/tt.jpg", "wb") as f:
#          f.write(decoded_bytes)
        img_buffer = io.BytesIO(decoded_bytes)
        img_np = np.frombuffer(img_buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return {"status":"error","message": str(e)}
 

  
