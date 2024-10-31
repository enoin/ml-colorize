import argparse

from nn.runner import ColorizeNNRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["test", "train"], help="Action to perform")
    args = parser.parse_args()
    runner = ColorizeNNRunner('assets/train', 'assets/test')
    match args.command:
        case "test":
            runner.test()
        case "train":
            runner.train()
        case _:
            print("Invalid command")


if __name__ == '__main__':
    main()
