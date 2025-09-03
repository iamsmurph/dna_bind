#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
#%%
with open("/data/rbg/users/ujp/dnabind/data/pbm/uniprobe/uniprobe_dataset.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
#%%
# Drop rows with negative intensity values from data
data = data[data.intensity >= 0]
intensities = data.intensity.values
log_intensities = np.log1p(intensities)

#%%
plt.hist(log_intensities, bins=500)
plt.title("PBM Intensity Distribution")
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.show()

# %%
