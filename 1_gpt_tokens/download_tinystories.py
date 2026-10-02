import os
import urllib.request

url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
folder = "./data/tinystories/"
filename = "input.txt"
path = os.path.join(folder, filename)


def main():
    os.makedirs(folder, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    print(f"Downloaded {filename} successfully.")


if __name__ == "__main__":
    main()
