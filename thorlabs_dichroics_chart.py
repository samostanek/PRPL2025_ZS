import matplotlib.pyplot as plt

# Mirror data (cut-on, reflection range, transmission range)
mirrors = [
    ("DMLP425", 425, (380, 410), (440, 800)),
    ("DMLP445", 445, (380, 430), (455, 800)),
    ("DMLP463", 463, (380, 453), (473, 800)),
    ("DMLP490", 490, (380, 475), (505, 800)),
    ("DMLP505", 505, (380, 490), (520, 800)),
    ("DMLP550", 550, (380, 533), (565, 800)),
    ("DMLP560", 560, (380, 547), (572, 950)),
    ("DMLP567", 567, (380, 550), (584, 800)),
    ("DMLP605", 605, (470, 590), (620, 800)),
    ("DMLP638", 638, (580, 621), (655, 800)),
]

# Spectral lines (wavelength, label)
spectral_lines = [
    (656.28, "Hα 656.28 nm"),
    (587.56, "He I 587.56 nm"),
    (447.14, "He I 447.14 nm"),
    (568, "N II 568 nm"),
    (479.46, "Cl II 479.46 nm"),
    (514.52, "C II 514.52 nm"),
    (777, "O II 777 nm"),
]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot reflection and transmission bands
for i, (name, cut_on, refl, trans) in enumerate(mirrors):
    y = len(mirrors) - i
    ax.plot(
        [cut_on, cut_on],
        [y - 0.3, y + 0.3],
        color="black",
        lw=1.5,
        label="Cut-on" if i == 0 else "",
    )
    ax.hlines(
        y,
        refl[0],
        refl[1],
        color="blue",
        lw=6,
        label="Reflection Band" if i == 0 else "",
    )
    ax.hlines(
        y,
        trans[0],
        trans[1],
        color="orange",
        lw=6,
        label="Transmission Band" if i == 0 else "",
    )
    ax.text(960, y, name, va="center", fontsize=9)

# Add spectral lines (labels below the x-axis)
for wl, label in spectral_lines:
    ax.axvline(wl, color="red", linestyle="--", lw=1)
    ax.text(
        wl, -1.75, label, rotation=90, va="top", ha="center", fontsize=8, color="red"
    )

# Labels and styling
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("Mirror", fontsize=12)
ax.set_title(
    "Thorlabs Long-Pass Dichroic Mirrors (<700 nm) with Spectral Lines",
    fontsize=14,
    weight="bold",
)
ax.set_yticks([])
ax.set_xlim(350, 1000)
ax.set_ylim(-1.5, len(mirrors) + 1)  # Extend space below axis
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# Remove the top and right borders (spines) and disable ticks there
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(top=False, right=False)

plt.tight_layout()
plt.show()
