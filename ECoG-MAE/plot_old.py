import pandas as pd
import numpy as np
import re
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


def plot_signal(args, dir):

    # read data
    fn = "/" + args.job_name + "_recon_signals.txt"
    df = pd.read_pickle(dir + fn)
    df["elec_numeric"] = df["elec"].apply(lambda x: int(re.search(r"\d+", x).group()))

    # plotting
    pdf_pages = PdfPages(dir + fn + ".pdf")
    elecs = df.groupby("elec_numeric")

    for key, elec in elecs:

        fig, axs = plt.subplots(5, 2, figsize=(10, 15))

        axs = axs.flatten()

        for e, ax in zip(elec.epoch, axs):

            ax.plot(
                np.linspace(0, len(elec.iloc[e, :].x[0]), len(elec.iloc[e, :].x[0])),
                elec.iloc[e, :].x[0],
                label="original",
                color="b",
                lw=0.25,
            )
            ax.plot(
                np.linspace(0, len(elec.iloc[e, :].y[0]), len(elec.iloc[e, :].y[0])),
                elec.iloc[e, :].y[0],
                label="reconstructed",
                color="r",
                lw=0.25,
            )

            ax.set_xlabel("datapoint in sample")
            ax.set_ylabel("highgamma power envelope")

            ax.legend()
            ax.set_title("epoch " + str(e + 1))

        plt.suptitle("Original and recon signal for elec G" + str(key))

        fig.tight_layout()

        pdf_pages.savefig(fig)
        plt.close(fig)

    # Close the PDF file
    pdf_pages.close()


def plot_corr(args, dir):

    print("plotting corr")

    # read data
    fn = "/" + args.job_name + "_train_corr.csv"
    df = pd.read_csv(dir + fn)
    df["elec_numeric"] = df["elec"].apply(lambda x: int(re.search(r"\d+", x).group()))
    groups = df.groupby(["elec", "band"])
    df["x"] = groups.cumcount()

    # plotting
    pdf_pages = PdfPages(dir + fn + ".pdf")

    colors = {"theta": "g", "alpha": "r", "beta": "b", "gamma": "c", "highgamma": "m"}

    elecs = df.groupby("elec_numeric")

    subplot_height_ratios = [1, 1, 1, 1, 1]

    # Iterate over each elec_numeric group
    for key, elec in elecs:
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=subplot_height_ratios)

        # Iterate over each electrode within the elec_numeric group
        for i, (_, band_group) in enumerate(elec.groupby("band")):
            ax = fig.add_subplot(gs[i, 0])

            # Plot 'corr' against 'x' for the band
            ax.plot(
                band_group["x"],
                band_group["corr"],
                label=f'{band_group["band"].iloc[0]}',
                color=colors[band_group["band"].iloc[0]],
                lw=0.25,
            )

            # Set title on the right side
            ax.set_title(f'{band_group["band"].iloc[0]}', loc="right")

            # Hide x ticks and labels on all but the bottom subplot
            if i < len(colors) - 1:
                ax.set_xticks([])
                ax.set_xticklabels([])

            # Set x axis label only on the bottom subplot
            if i == len(colors) - 1:
                ax.set_xlabel("Iteration")

            ax.set_ylabel("Corr (r)")

            # ax.legend()

        # Set title for the entire PDF page
        fig.suptitle(f"G{key}", fontsize=16)

        # Adjust layout and save the subplot to the PDF file
        fig.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Add a little space at the top for the title
        pdf_pages.savefig(fig)
        plt.close(fig)

    # Close the PDF file
    pdf_pages.close()


parser = argparse.ArgumentParser()
parser.add_argument("--job-name", type=str)
parser.add_argument("--plot-type", type=str)
args = parser.parse_args()

cwd = os.getcwd()
dir = os.path.dirname(cwd) + "/ECoG-foundation-model/results"

print(dir)

if args.plot_type == "corr":
    plot_corr(args, dir)
    print("corr")

elif args.plot_type == "signal":
    plot_signal(args, dir)
