from pathlib import Path

import gdown

URLS = {
    "https://drive.google.com/uc?id=1EVukddCkHn2V38IElnWnZ-FrEVanjeo8": "saved/rtfs_net.pth",
    "https://drive.google.com/uc?id=1viqbvF_GOkSw2T7hZPCfaDo0fxR6hsSe": "saved/artfs_net.pth",
    "https://drive.google.com/uc?id=13M1g8N8HkquSDwZb-zkWPTH52wyoK85f": "saved/final_model.pth",
    "https://drive.google.com/uc?id=1ojSZd6-VqAkcvyT1RYpCgvM53tNmIL_U": "saved/convtasnet.pth",
    "https://drive.google.com/uc?id=1HuJCj1KotKTTbiEgDBMk5lQ5Df0gDBXC": "saved/sepreformer_t.pth",
}


def main():
    path_gzip = Path("saved/").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    for url, path in URLS.items():
        gdown.download(url, path)
        print("Model downloaded to", str(Path(path).absolute().resolve()))


if __name__ == "__main__":
    main()
