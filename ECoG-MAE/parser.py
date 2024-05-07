import argparse
import ast


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.set_defaults(debug=False)
    parser.add_argument("--data-size", type=float, default=1)
    parser.add_argument("--sandbox", dest="sandbox", action="store_true")
    parser.set_defaults(sandbox=False)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--new-fs", type=int)
    parser.add_argument("--sample-length", type=int)
    parser.add_argument("--patch-size", nargs="+", type=int)
    parser.add_argument("--frame-patch-size", type=int)
    parser.add_argument("--tube-mask-ratio", type=float)
    parser.add_argument("--decoder-mask-ratio", type=float)
    parser.add_argument(
        "--running-cell-masking", dest="running_cell_masking", action="store_true"
    )
    parser.set_defaults(running_cell_masking=False)
    parser.add_argument("--bands", type=str)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--use-cls-token", dest="use_cls_token", action="store_true")
    parser.set_defaults(use_cls_token=False)
    parser.add_argument(
        "--use-contrastive-loss", dest="use_contrastive_loss", action="store_true"
    )
    parser.set_defaults(use_contrastive_loss=False)
    args = parser.parse_args()

    # parse string input to list of lists
    args.bands = ast.literal_eval(args.bands)

    return args
