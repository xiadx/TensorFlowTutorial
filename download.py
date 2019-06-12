#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Function for downloading and extracting data-files from the internet"""


import sys
import os
import urllib.request
import zipfile
import tarfile


def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """
    # Percentage complete
    pct_complete = float(count * block_size) / total_size

    # Limit it because rounding errors may cause it to exceed 100%
    pct_complete = min(1.0, pct_complete)

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def download(base_url, filename, download_dir):
    """
    Download the given file if it does not already exist in the download_dir.
    Args:
        base_url: The internet URL without the filename.
        filename: The filename that will be added to the base_url.
        download_dir: Local directory for storing the file
    Return:
         Nothing.
    """
    # Path for local file.
    save_path = os.path.join(download_dir, filename)

    # Check if the file already exists, otherwise we need to download it now.
    if not os.path.exists(save_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        print("Downloading", filename, "...")

        # Downloading the file from the internet
        url = base_url + filename
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=save_path,
                                                  reporthook=_print_download_progress)

        print(" Done!")


def maybe_download_and_extract(url, download_dir):
    """
    Download and extract the data if it doesn't already exists.
    Assumes the url is a tar-ball file.
    Args:
        url: Internet URL for the tar-file to download.
             Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        download_dir: Directory where the downloaded file is saved.
                      Example: "data/CIFAR-10/"
    Return:
        Nothing.
    """
    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path,
                                                  reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


def main():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_dir = "data/CIFAR-10/"
    maybe_download_and_extract(url, download_dir)


if __name__ == "__main__":
    main()