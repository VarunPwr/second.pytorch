import argparse


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--task-name", type=str, default="torques_from_contact_forces")
    arg_parser.add_argument("--use-additional-info", action="store_true", default=False)
    args = arg_parser.parse_args()

    return args
