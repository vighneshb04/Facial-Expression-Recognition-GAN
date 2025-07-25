import numpy as np
import matplotlib.pyplot as plt
import os

# Change to your actual path
folder = "output_landmarks"

# Load a sample .npy file
sample_file = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])[0]
data = np.load(os.path.join(folder, sample_file))

# Reshape if necessary
if data.shape[0] == 68 and data.shape[1] == 2:
    x = data[:, 0]
    y = data[:, 1]
else:
    print("Unexpected shape:", data.shape)
    exit()

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(x, -y, c='blue')  # flip y to match face orientation
for i in range(68):
    plt.text(x[i], -y[i], str(i), fontsize=8)
plt.title(f"Facial Landmarks from: {sample_file}")
plt.show()
