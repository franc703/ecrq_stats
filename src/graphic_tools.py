import matplotlib.pyplot as plt

def scatter_plot(x, y, x_label='X', y_label='Y', title='Scatter Plot', color='blue', alpha=0.5, size=10):
    """
    Plots a scatter plot for two variables.

    :param x: List or array of values for the X-axis
    :param y: List or array of values for the Y-axis
    :param x_label: Label for the X-axis (default 'X')
    :param y_label: Label for the Y-axis (default 'Y')
    :param title: Title for the plot (default 'Scatter Plot')
    :param color: Color of the points (default 'blue')
    :param alpha: Transparency of the points (default 0.5)
    :param size: Size of the points (default 10)
    """
    plt.scatter(x, y, c=color, alpha=alpha, s=size)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()
