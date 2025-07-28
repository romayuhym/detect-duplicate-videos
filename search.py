import faiss, pickle

from gpt_version import get_video_embedding

top_k = 10

if __name__ == "__main__":
    index = faiss.read_index("video_index.faiss")

    with open("video_paths.pkl", "rb") as f:
        video_paths = pickle.load(f)


    query_path = "samples/Lasta_Bikini beach_3/Lasta_Bikini beach_3-3.mov"
    print(f"Пошук схожих відео для: {query_path}")
    query_emb = get_video_embedding(query_path).reshape(1, -1)
    D, I = index.search(query_emb, top_k)

    print("Топ схожих відео:")
    for score, idx in zip(D[0], I[0]):
        print(f"{video_paths[idx]} \tСхожість: {score:.4f}")
