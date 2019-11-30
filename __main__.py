import argparse

from .fine_tune_model import fine_tune
#from .tokenization import tokenize

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_name')
    parser.add_argument('train_corpus')
    parser.add_argument('dev_corpus')
    parser.add_argument('--columns', nargs='*')
    args = parser.parse_args()

    print(args)

    fine_tune(args.corpus_name,
              args.train_corpus,
              args.dev_corpus,
              args.column_names)
    #tokenize(args.bagus)
