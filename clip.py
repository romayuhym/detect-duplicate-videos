import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytesseract
from pytesseract import Output
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# опційно обмежте кількість потоків
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
from PIL import Image
import torch
import faiss
import numpy as np
import torch.nn.functional as F
from open_clip import create_model_from_pretrained

# Параметри
num_frames = 16
top_k = 15
# Порогова схожість
threshold = 0.8

video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

# Модель OpenVision
model, preprocess = create_model_from_pretrained(
    "hf-hub:UCSC-VLAA/openvision-vit-base-patch8-224",
    device="mps")


def load_videos() -> list:
    video_list = []
    for root, dirs, files in os.walk('samples'):
        for file in files:
            if file.endswith(video_extensions):
                video_list.append(os.path.join(root, file))
    return video_list



def mask_text_pytesseract(img_bgr: np.ndarray, conf_threshold: int = 60) -> np.ndarray:
    """
    Виявляє текст на зображенні через pytesseract і замальовує відповідні області.
    :param img_bgr: вхідний кадр у форматі BGR (NumPy array)
    :param conf_threshold: мінімальна довіра OCR для маскування
    :return: кадр із замаскованим текстом
    """
    # Отримуємо дані OCR
    data = pytesseract.image_to_data(img_bgr, config='--oem 3 --psm 6', output_type=Output.DICT)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > conf_threshold:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Замальовуємо прямокутник (чорно)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return img_bgr


def extract_frames(path, num_frames=num_frames):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(20, max(total - 20, 0), num_frames, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        frames.append(frame_bgr)

        # frame_masked = mask_text_pytesseract(frame_bgr)
        # rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
        # frames.append(Image.fromarray(rgb))
    cap.release()
    # return frames

    # with ThreadPoolExecutor(max_workers=4) as ex:  # 4 потоки
    #     frames = list(ex.map(mask_text_pytesseract, frames))

    # Конвертуємо в PIL
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def embed_frames(frames):
    imgs = torch.stack([preprocess(img) for img in frames]).to('mps')
    with torch.no_grad():
        feats = model.encode_image(imgs)
    return F.normalize(feats, dim=-1)


def get_video_embedding(path):
    frames = extract_frames(path)
    feats = embed_frames(frames)
    # Усереднюємо і нормалізуємо
    mean_feat = feats.mean(dim=0, keepdim=True)  # (1, D)
    mean_feat = F.normalize(mean_feat, dim=-1)  # (1, D)
    return mean_feat.squeeze(0).cpu().numpy().astype("float32")


if __name__ == "__main__":
    # Індексування відео
    video_paths = load_videos()
    embeddings = []
    for video_path in tqdm(video_paths, desc="Створення ембеддінгів відео: "):
        emb = get_video_embedding(video_path)
        embeddings.append(emb)

    video_embeddings = np.array(embeddings)
    video_embeddings = video_embeddings.astype('float32')
    print("embedding завершено, розмір:", video_embeddings.shape)

    dim = video_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(video_embeddings)

    faiss.write_index(index, "video_index.faiss")
    with open("video_paths.pkl", "wb") as f:
        pickle.dump(video_paths, f)

# # Пошук схожих відео
# query_path = "samples/Lasta_Bikini beach_3/Lasta_Bikini beach_3-3.mov"
# print(f"Пошук схожих відео для: {query_path}")
# query_emb = get_video_embedding(query_path).reshape(1, -1)
# D, I = index.search(query_emb, top_k)
#
# print("Топ схожих відео:")
# for score, idx in zip(D[0], I[0]):
#     print(f"{video_paths[idx]} \tСхожість: {score:.4f}")



# query_path = "samples/Lasta_Bikini beach_3/Lasta_Bikini beach_3-3.mov"
# query_emb = get_video_embedding(query_path).reshape(1, -1)
# q = query_emb.astype('float32').reshape(1, -1)
# lims, distances, indices = index.range_search(q, threshold)
# start, end = lims[0], lims[1]
#
# print(f"Знайдено {end-start} відео зі схожістю ≥ {threshold}:")
# for pos in range(start, end):
#     vid_idx = indices[pos]
#     score = distances[pos]
#     print(f"  {video_paths[vid_idx]} — Схожість: {score:.4f}")


# res = {}
# for idx, vid in enumerate(embeddings):
#     q = vid.astype('float32').reshape(1, -1)
#     lims, distances, indices = index.range_search(q, threshold)
#     start, end = lims[0], lims[1]
#     for pos in range(start, end):
#         name = video_paths[idx]
#         if name not in res:
#             res[name] = []
#         res[name].append(video_paths[indices[pos]])
#
# print(res)
