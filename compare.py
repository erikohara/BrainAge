"""
Checks if any of Erik and Vibu's subjects are diseased using Beth's filtering code.
@author: Finn
"""

import os

print("Fetching Beth data...")
f1 = [name for name in os.listdir("/work/forkert_lab/elizabeth/neuro")]
neuro = [name[4:] for name in f1 if "sub-" in name]

print("Fetching Erik/Vibu data...")
f2 = [name for name in os.listdir("/work/forkert_lab/erik/T1_cropped_slices/T1_cropped_slice_91")]
data = [name[:7] for name in f2 if ".tiff" in name]

# print(neuro)
# print(data)

overlap = [name for name in data if name in neuro]
# print(overlap)

print(f"{len(overlap)} out of {len(data)} subjects ({len(overlap)/len(data)*100}%) in the data are diseased.")

with open("overlap.txt", "w") as file:
    print("Saving overlapping subjects to 'overlap.txt'...")

    for name in overlap:
        file.write("%s\n" % name)
    print("Done.")