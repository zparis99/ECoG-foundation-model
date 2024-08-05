"""Code adapted from: https://github.com/hassonlab/247-encoding/blob/e0b7468824bc950f15dd8e47f9b7c4bdb3615109/scripts/tfsenc_load_signal.py"""

import glob
import os

import numpy as np
from scipy import stats
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# def trim_signal(signal):
#     bin_size = 32  # 62.5 ms (62.5/1000 * 512)
#     signal_length = signal.shape[0]

#     if signal_length < bin_size:
#         print("Ignoring conversation: Small signal")
#         return None

#     cutoff_portion = signal_length % bin_size
#     if cutoff_portion:
#         signal = signal[:-cutoff_portion, :]

#     return signal


def detrend_signal(mat_signal):  # Detrending
    """Detrends a signal

    Args:
        mat_signal: signal for a specific conversation

    Returns:
        mat_signal: detrended signal
    """

    y = mat_signal
    X = np.arange(len(y)).reshape(-1, 1)
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X)

    model = LinearRegression()
    model.fit(Xp, y)
    trend = model.predict(Xp)
    mat_signal = y - trend

    return mat_signal


def create_nan_signal(stitch, convo_id):
    """Returns fake signal for a conversation

    Args:
        stitch: stitch_index
        convo_id: conversation id

    Returns:
        mat_signal: nans of a specific conversation size
    """

    mat_len = stitch[convo_id] - stitch[convo_id - 1]  # mat file length
    mat_signal = np.empty((mat_len, 1))
    mat_signal.fill(np.nan)

    return mat_signal


def load_electrode_data(args, conversation_ids: list[int], sid: int, elec_id, stitch=None, z_score=False):
    """Load and concat signal mat files for a specific electrode

    Args:
        args (namespace): commandline arguments
        conversation_ids: list of conversation ids to load data for
        elec_id: electrode id
        stitch: stitch_index
        z_score: whether we z-score the signal per conversation

    Returns:
        elec_signal: concatenated signal for a specific electrode
        elec_datum: modified datum based on the electrode signal
    """

    all_signal = []
    missing_convos = []
    for convo_id in conversation_ids:
        if args.conversation_id != 0 and convo_id != args.conversation_id:
            continue

        file = glob.glob(
            args.electrode_data_path.format(convo_id=convo_id, sid=sid, elec_id=elec_id)
        )

        if len(file) == 1:  # conversation file exists
            file = file[0]

            mat_signal = loadmat(file)["p1st"]
            mat_signal = mat_signal.reshape(-1, 1)

            if mat_signal is None:
                continue

            mat_signal = detrend_signal(mat_signal)  # detrend conversation signal
            if z_score:  # doing erp
                mat_signal = stats.zscore(mat_signal)

        elif len(file) == 0:  # conversation file does not exist
            if not stitch:
                raise ValueError(f"Electrode file does not exist for sid: {sid} and conversation id: {convo_id} and stitch not provided to create nan signal.")
            # if args.sid != 7170:
            #     raise SystemExit(
            #         f"Error: Conversation file does not exist for electrode {elec_id} at {convo}"
            #     )
            missing_convos.append(convo_id)  # append missing convo name
            mat_signal = create_nan_signal(stitch, convo_id)

        else:  # more than 1 conversation files
            raise SystemExit(
                f"Error: More than 1 signal file exists for electrode {elec_id} at {args.electrode_data_path}"
            )

        all_signal.append(mat_signal)  # append conversation signal

    if args.project_id == "tfs":
        elec_signal = np.vstack(all_signal)
    else:
        elec_signal = np.array(all_signal)

    return elec_signal, missing_convos
