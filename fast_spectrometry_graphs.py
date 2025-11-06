import time
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import numpy as np

# Increase global font sizes for readability
plt.rcParams.update(
    {
        "font.size": 12,  # base font size
        "axes.labelsize": 18,  # x/y labels
        "axes.titlesize": 18,  # subplot titles
        "xtick.labelsize": 14,  # x tick labels
        "ytick.labelsize": 14,  # y tick labels
        "legend.fontsize": 14,  # legends
        "figure.titlesize": 16,  # suptitle
    }
)

shotno = 50389

# adjust base URL or file-path as needed
base_url = f"http://golem.fjfi.cvut.cz/shots/{shotno}/Diagnostics/FastSpectrometry/"

files = {
    "Hα": "U_Halpha.csv",
    "Hβ": "U_Hbeta.csv",
    "He I": "U_HeI.csv",
    "Whole": "U_whole.csv",
}

data = {}
while True:
    try:
        for label, fname in files.items():
            url = base_url + fname
            # assume comma delimiter; skip header lines if needed
            df = pd.read_csv(url)
            data[label] = df
            print(f"{label}: loaded {df.shape[0]} rows, columns = {list(df.columns)}")
        break  # exit loop if successful
    except urllib.error.HTTPError as e:
        print(f"Failed to load data for shot {shotno}, retrying...")
        time.sleep(2)
        continue


# One figure, four stacked subplots (share X for alignment)
fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(12, 10), sharey=True)

noise = np.roll(data["Whole"], -2)

for ax, (label, df) in zip(axes, data.items()):
    t = df.iloc[:, 0] * 1e3  # convert to ms
    u = df.iloc[:, 1]
    ax.plot(t, u, label=label, marker=".")
    ax.plot(t, noise[:, 0], color="gray", alpha=0.5, label="Noise (Whole)", marker=".")
    # Use LaTeX-style mathtext for axis label
    ax.set_ylabel(r"$I$ [a.u.]")
    ax.grid(True)
    ax.set_title(label)

# Label only the bottom axis with X label
# Use LaTeX-style mathtext for x-axis label
axes[-1].set_xlabel(r"$t$ [ms]")

# Figure-level title (fontsize controlled by rcParams figure.titlesize)
# Move it slightly closer to the subplots
fig.suptitle(f"Fast Spectrometry Signals - Shot {shotno}", y=0.975)

# Adjust layout; raise the top of the axes area to sit closer to the suptitle
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

# Show the single figure with all four subplots
plt.show()


# offsets = np.arange(-100, 100)
# correlations = []
# for o in offsets:
#     noise_np = np.array(noise.iloc[:, 1])
#     data_np = np.array(noise.iloc[:, 1])
#     shifted = np.roll(noise_np, o)
#     corr = np.corrcoef(data_np, shifted)[0, 1]
#     correlations.append(corr)

# plt.figure(figsize=(10, 5))
# plt.plot(offsets, correlations, marker="o")
# plt.show()
