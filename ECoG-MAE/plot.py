import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


def plot_signal(args, signal, id):

    plt.figure(figsize=(8, 3))
    plt.plot(signal)
    plt.title("Training signal")
    dir = os.getcwd() + f"/results/signals/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + f"{args.job_name}_{id}_signal.png")


def plot_signal_stats(args, signal_means, signal_stds):

    plt.figure(figsize=(8, 3))
    plt.plot(signal_means)
    plt.title("Training signal means")
    dir = os.getcwd() + f"/results/signals/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + f"{args.job_name}_signal_means.png")

    plt.figure(figsize=(8, 3))
    plt.plot(signal_stds)
    plt.title("Training signal stds")
    dir = os.getcwd() + f"/results/signals/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + f"{args.job_name}_signal_stds.png")


def plot_losses(args, train_losses, seen_train_losses, test_losses, seen_test_losses):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Recon training losses")

    axs[0, 1].plot(seen_train_losses)
    axs[0, 1].set_title("Seen training losses")

    axs[1, 0].plot(test_losses)
    axs[1, 0].set_title("Recon test losses")

    axs[1, 1].plot(seen_test_losses)
    axs[1, 1].set_title("Seen test losses")

    plt.tight_layout()

    dir = os.getcwd() + f"/results/loss/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + f"{args.job_name}_losses.png")


def plot_contrastive_loss(args, contrastive_losses):
    plt.figure(figsize=(8, 3))
    plt.plot(contrastive_losses)
    plt.title("Training contrastive losses")
    dir = os.getcwd() + f"/results/loss/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + f"{args.job_name}_contrastive_loss.png")


def plot_correlation(args, df, fn):

    dir = os.getcwd() + f"/results/correlation/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    groups = df.groupby(["elec", "band"])
    # test to do df.dl_i instead #TODO
    df["x"] = groups.cumcount()

    # plotting
    pdf_pages = PdfPages(dir + f"{args.job_name}_{fn}.pdf")

    colors = {"theta": "g", "alpha": "r", "beta": "b", "gamma": "c", "highgamma": "m"}

    elecs = df.groupby("elec")

    subplot_height_ratios = [1, 1, 1, 1, 1]

    # Iterate over each elec group
    for key, elec in elecs:
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=subplot_height_ratios)

        # Iterate over each band within the elec group
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


def plot_recon_signals(args, df):

    dir = os.getcwd() + f"/results/recon_signals/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # plotting
    pdf_pages = PdfPages(dir + f"{args.job_name}_recon_signals.pdf")
    elecs = df.groupby("elec")

    for key, elec in elecs:

        fig, axs = plt.subplots(int(np.ceil(args.num_epochs / 2)), 2, figsize=(10, 15))

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
