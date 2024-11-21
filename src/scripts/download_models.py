from pathlib import Path

import gdown

URLS = {
    "https://drive.google.com/uc?id=1EVukddCkHn2V38IElnWnZ-FrEVanjeo8": "saved/rtfs_net.pth",
    "https://drive.google.com/uc?id=1viqbvF_GOkSw2T7hZPCfaDo0fxR6hsSe": "saved/artfs_net.pth",
    "https://drive.google.com/uc?id=1AAcxbO6_W82t9uLpVPa6YZlALf75aPbl": "saved/final_model.pth",
}


def main():
    path_gzip = Path("saved/").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    for url, path in URLS.items():
        gdown.download(url, path)
        print("Model downloaded to", str(Path(path).absolute().resolve()))


if __name__ == "__main__":
    main()
