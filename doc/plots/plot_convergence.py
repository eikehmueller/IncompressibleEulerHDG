import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

from matplotlib import pyplot as plt

# velocity errors on grids of size 2x2, 4x4, 8x8, 16x16, 32x32
# polynomial degrees are:
#  * p=1 for implicit and ARS2(2,3,2)
#  * p=2 for implicit and SSP3(4,3,3)

v_error = {
    "implicit": [
        0.33861601952691356,
        0.20952447183370812,
        0.12450270079447295,
        0.0711728199304723,
        0.038670387558933095,
    ],
    "ars232": [
        0.03972723338605709,
        0.00813957483678574,
        0.00119820072423085,
        0.00015379731856933536,
        2.07664410221023e-05,
    ],
    "ssp433": [
        0.01497743258664723,
        0.0011548499368503058,
        0.00010117767353696738,
        9.794825258167717e-06,
        1.0981937627693843e-06,
    ],
}

color = {"implicit": "blue", "ars232": "red", "ssp433": "green"}
marker = {"implicit": "o", "ars232": "s", "ssp433": "v"}
label = {
    "implicit": "implicit, $p=1$",
    "ars232": "ARS2(2,3,2), $p=1$",
    "ssp433": "SSP3(4,3,3), $p=2$",
}

ndata = max([len(data) for data in v_error.values()])

plt.clf()
ax = plt.gca()
ax.set_xticks(range(ndata))
ax.set_ylim(5e-7, 1)
ax.set_xticklabels([f"${2**(j+1):d}\\times{2**(j+1):d}$" for j in range(ndata)])
ax.set_xlabel("mesh")
ax.set_ylabel("Velocity error $||Q-Q_{\\operatorname{exact}}||_2$")
ax.set_yscale("log")
for timestepper, data in v_error.items():
    plt.plot(
        data,
        linewidth=2,
        color=color[timestepper],
        label=label[timestepper],
        markersize=6,
        marker=marker[timestepper],
        markeredgewidth=2,
        markerfacecolor="white",
    )

plt.plot([1, 3], [0.5, 0.5 / 2**2], linewidth=2, color="black", linestyle="--")
plt.annotate("$\\propto h$", (2, 0.4))
plt.plot([1, 3], [3.0e-2, 3.0e-2 / 4**2], linewidth=2, color="black", linestyle="--")
plt.annotate("$\\propto h^2$", (2, 1.0e-2))
plt.plot([1, 3], [3.0e-3, 3.0e-3 / 8**2], linewidth=2, color="black", linestyle="--")
plt.annotate("$\\propto h^3$", (3, 2.0e-5))

plt.legend(loc="lower left")
plt.savefig("convergence.pdf", bbox_inches="tight")
