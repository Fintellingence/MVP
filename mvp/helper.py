import matplotlib.pyplot as plt


def get_symbols(file):
    with open(file) as f:
        tickers = f.readlines()
    stripper = lambda x: (x.strip())
    return list(map(stripper, tickers))


def plot_two_series(series_a, series_b):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Series A", color=color)
    ax1.plot(series_a, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#0E0C0C"
    ax2.set_ylabel("Series B", color=color)
    ax2.plot(series_b, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return None
