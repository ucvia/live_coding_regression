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

def plot_regularization_path(lambd_values, beta_values, true_features, legend="\lambda", title="Regularization Path"):
    """Method to plot the regularization path given lambda values and beta values
    Args:
        lambd_values (numpy.ndarray): An array of lambda values
        beta_values (numpy.ndarray): A matrix of size (n, lambdas) with n the number of lambdas.
        legend (str): The x-axis label of interest (Defaults to lambda).
        title (str): The plot title name (Defaults to Regularization Path)
    """
    num_coeffs = len(beta_values[0])
    plt.figure(figsize=(10,10))
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values], linewidth=3 if i in true_features else 1)
    plt.xlabel(r"${}$".format(legend), fontsize=16)
    plt.xscale("log")
    plt.title("{}".format(title), fontsize=20)
    # plt.show()
    return plt