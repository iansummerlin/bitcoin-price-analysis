import os
import requests

def print_divider():
    """Print a divider to improve organisation of print statements."""
    print("------------------------")

def download_file(url, path):
    """
    Download a file from a URL if it doesn't already exist.

    Parameters:
    url (str): The URL to download the file from.
    path (str): The local path where the file will be saved.
    """
    print_divider()
    print(f"Downloading file from {url}...")
    if os.path.exists(path):
        print(f"The file '{path}' already exists.")
        return
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the file. Status code: {response.status_code}")
        return
    
    with open(path, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded and saved as {path}")