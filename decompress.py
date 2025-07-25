import bz2
import urllib.request
import os

# Download and decompress landmark file
url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
filename = "shape_predictor_68_face_landmarks.dat.bz2"

print("Downloading landmark file...")
urllib.request.urlretrieve(url, filename)

print("Decompressing...")
with bz2.BZ2File(filename, "rb") as f_in:
    with open("shape_predictor_68_face_landmarks.dat", "wb") as f_out:
        f_out.write(f_in.read())

print("Done! Cleanup...")
os.remove(filename)
