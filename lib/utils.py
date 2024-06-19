import matplotlib.pyplot as plt

def plot_coefficients(df, w=20, h=10):
    """[summary]
    Args:
        plot_info ([type]): [description]
    """
    cmaps = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    plot_info = {
        k: {
            "color": cmaps[i%len(cmaps)], "size": 4 + i*4, "w": w.values
        } for i, (k, w) in enumerate(df.items())
    }

    plt.figure(figsize=(w, h))
    for label, value in plot_info.items():
        plt.scatter(
            range(len(value["w"])),
            value["w"],
            label=label,
            c=value["color"],
            s=value["size"]
        )
    # plt.text(0.0, -1.5, r"$\lambda={:3}$".format(Lambda))
    plt.title("Value of coefficients per solution")
    plt.ylabel("value")
    plt.xlabel(r"Coefficient $i$")
    plt.legend()
    # plt.show()
    return plt