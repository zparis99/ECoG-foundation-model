import argparse
import ast


def arg_parser():
    parser = argparse.ArgumentParser()
    # General config
    parser.add_argument("--config-file", type=str, default="configs/video_mae_train.ini")
    parser.add_argument("--job-name", type=str, help="Name of training job. Will be used to save metrics.")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.set_defaults(debug=False)

    # ViTConfig parameters
    parser.add_argument("--dim", type=int, 
                       help="Dimensionality of token embeddings.")
    parser.add_argument("--decoder-embed-dim", type=int,
                       help="Dimensionality to transform encoder embeddings into when passing into the decoder.")
    parser.add_argument("--mlp-ratio", type=float,
                       help="Ratio of input dimensionality to use as a hidden layer in Transformer Block MLP's")
    parser.add_argument("--depth", type=int,
                       help="Depth of encoder.")
    parser.add_argument("--decoder-depth", type=int,
                       help="Depth of decoder.")
    parser.add_argument("--num-heads", type=int,
                       help="Number of heads in encoder.")
    parser.add_argument("--decoder-num-heads", type=int,
                       help="Number of heads in decoder.")
    parser.add_argument("--patch-size", type=int,
                       help="The number of electrodes in a patch.")
    parser.add_argument("--frame-patch-size", type=int,
                       help="The number of frames to include in a tube per video mae.")
    parser.add_argument("--use-cls-token", dest="use_cls_token", action="store_true",
                       help="Prepend classification token to input if True.")
    parser.set_defaults(use_cls_token=False)
    parser.add_argument("--sep-pos-embed", dest="sep_pos_embed", action="store_true",
                       help="If true then use a separate position embedding for the decoder.")
    parser.set_defaults(sep_pos_embed=False)
    parser.add_argument("--trunc-init", dest="trunc_init", action="store_true",
                       help="Use truncated normal initialization if True.")
    parser.set_defaults(trunc_init=False)
    parser.add_argument("--no-qkv-bias", dest="no_qkv_bias", action="store_true",
                       help="If True then don't use a bias for query, key, and values in attention blocks.")
    parser.set_defaults(no_qkv_bias=False)

    # VideoMAETaskConfig parameters
    parser.add_argument("--encoder-mask-ratio", type=float,
                       help="Proportion of tubes to mask out. See VideoMAE paper for details.")
    parser.add_argument("--decoder-mask-ratio", type=float,
                       help="The ratio of the number of masked tokens in the input sequence.")
    parser.add_argument("--norm-pix-loss", dest="norm_pix_loss", action="store_true",
                       help="If true then normalize the target before calculating loss.")
    parser.set_defaults(norm_pix_loss=False)

    # ECoGDataConfig parameters
    parser.add_argument("--norm", type=str,
                       help="If 'batch' then will normalize data within a batch.")
    parser.add_argument("--data-size", type=float,
                       help="Percentage of data to include in training/testing.")
    parser.add_argument("--batch-size", type=int,
                       help="Batch size to train with.")
    parser.add_argument("--env", dest="env", action="store_true",
                       help="If true then convert data to power envelope by taking magnitude of Hilbert transform.")
    parser.set_defaults(env=False)
    parser.add_argument("--bands", type=str,
                       help="Frequency bands for filtering raw iEEG data.")
    parser.add_argument("--original-fs", type=int,
                       help="Original sampling frequency of data.")
    parser.add_argument("--new-fs", type=int,
                       help="Frequency to resample data to.")
    parser.add_argument("--dataset-path", type=str,
                       help="Relative path to the dataset root directory.")
    parser.add_argument("--train-data-proportion", type=float,
                       help="Proportion of data to have in training set. The rest will go to test set.")
    parser.add_argument("--sample-length", type=int,
                       help="Number of seconds of data to use for a training example.")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true",
                       help="If true then shuffle the data before splitting to train and eval.")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--test-loader", dest="test_loader", action="store_true",
                       help="If True then uses a mock data loader.")
    parser.set_defaults(test_loader=False)

    # TrainerConfig parameters
    parser.add_argument("--max-learning-rate", type=float,
                       help="Max learning rate for scheduler.")
    parser.add_argument("--num-epochs", type=int,
                       help="Number of epochs to train over data.")
    parser.add_argument("--loss", type=str,
                       help="Type of loss to use.")

    # LoggingConfig parameters
    parser.add_argument("--event-log-dir", type=str,
                       help="Directory to write logs to (i.e. tensorboard events, etc).")
    parser.add_argument("--print-freq", type=int,
                       help="Number of steps to print training progress after.")

    args = parser.parse_args()

    # Parse string input to list of lists
    args.bands = ast.literal_eval(args.bands) if args.bands else None

    return args
