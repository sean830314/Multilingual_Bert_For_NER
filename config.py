"""
Config Parser
"""
import inspect
import os

import configargparse


def get_args():
    """
    get args as config
    """
    args, _ = parse().parse_known_args()
    return args


def parse():
    """
    adds and parses arguments / hyperparameters
    """
    default = os.path.join(current(), "config.yml")
    parser = configargparse.ArgumentParser(default_config_files=[default])
    parser.add("-c", "--config-file", is_config_file=True, help="config file path")
    parser.add_argument(
        "--do_train", default=False, action="store_true", help="Run training stage"
    )
    parser.add_argument(
        "--do_eval", default=False, action="store_true", help="Run evaluation stage"
    )
    parser.add_argument(
        "--do_predict", default=False, action="store_true", help="Run prediction stage"
    )
    parser.add_argument(
        "--output_model_dir",
        type=str,
        help="model path of train or eval",
    )
    parser.add_argument(
        "--use_model",
        default="bert-base-multilingual-cased",
        type=str,
        help="name of use model, and default is xlm-roberta-base or bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--tokenizer",
        default="bert-base-multilingual-cased",
        type=str,
        help="name of use tokenizer, and default is xlm-roberta-base or bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning rate for train process",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=800,
        help="step size for train process",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="language of dataset",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="stride size for sliding window algorithm on train process",
    )
    return parser


def current():
    """
    returns the current directory path
    """
    path = os.path.abspath(path=inspect.getfile(inspect.currentframe()))
    head, _ = os.path.split(path)
    return head


if __name__ == "__main__":
    arg = get_args()
    print(arg)
