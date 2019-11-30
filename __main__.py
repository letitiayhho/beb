import argparse

from .tokenization import tokenize

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print(args)

    #tokenize(args.bagus)
