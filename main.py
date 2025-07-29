import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import click
import numpy as np
import cv2
import faiss
from PIL import Image
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained

video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


def load_videos(path_to_video_folder) -> list:
    video_list = []
    for root, dirs, files in os.walk(path_to_video_folder):
        for file in files:
            if file.endswith(video_extensions):
                video_list.append(os.path.join(root, file))
    return video_list


def extract_frames(path, num_frames):
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

    cap.release()

    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]


def embed_frames(frames, model, preprocess):
    imgs = torch.stack([preprocess(img) for img in frames]).to('mps')
    with torch.no_grad():
        feats = model.encode_image(imgs)
    return F.normalize(feats, dim=-1)


def get_video_embedding(path, model, preprocess, num_frames):
    frames = extract_frames(path, num_frames)
    feats = embed_frames(frames, model, preprocess)
    mean_feat = feats.mean(dim=0, keepdim=True)  # (1, D)
    mean_feat = F.normalize(mean_feat, dim=-1)  # (1, D)
    return mean_feat.squeeze(0).cpu().numpy().astype("float32")


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


@click.command()
@click.option('--path', default='samples', help='Path to the folder with videos')
@click.option('--num_frames', default=16, help='Number of frames to extract from each video', type=int)
@click.option('--threshold', default=0.8, help='Similarity threshold for duplicate detection', type=float)
def main(path, num_frames, threshold):
    click.echo('Collect video')
    video_paths = load_videos(path)
    click.echo('Found {} videos'.format(len(video_paths)))
    if not video_paths:
        click.echo('No videos found in the specified folder.')
        return

    click.echo('Load OpenVision model')
    model, preprocess = create_model_from_pretrained(
        "hf-hub:UCSC-VLAA/openvision-vit-base-patch8-224",
        device=device
    )
    click.echo('Model loaded successfully')
    click.echo('Start embedding videos')
    embeddings = []
    with click.progressbar(video_paths, label="Creation of video embeddings") as bar:
        for video_path in bar:
            emb = get_video_embedding(video_path, model, preprocess, num_frames)
            embeddings.append(emb)

    video_embeddings = np.array(embeddings)
    video_embeddings = video_embeddings.astype('float32')
    click.echo("Embedding completed")

    dim = video_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(video_embeddings)

    duplicates = []
    create_directory("results")
    folder_idx = 1
    with click.progressbar(video_paths, label="Search for duplicates") as bar:
        for i, video_path in enumerate(bar):
            if video_path in duplicates:
                continue

            create_directory("results/{}".format(folder_idx))

            q = video_embeddings[i].reshape(1, -1)
            lims, distances, indices = index.range_search(q, threshold)
            start, end = lims[0], lims[1]

            for pos in range(start, end):
                vid_idx = indices[pos]
                duplicate_video_path = video_paths[vid_idx]

                if duplicate_video_path in duplicates:
                    continue

                duplicates.append(duplicate_video_path)
                # Copy the video to the results folder
                os.system(f'cp "{duplicate_video_path}" "results/{folder_idx}/"')

            folder_idx += 1

    click.echo("All videos processed successfully")


if __name__ == '__main__':
    main()
