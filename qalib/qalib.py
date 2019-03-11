#!/usr/bin/python3

import sys
import argparse

from .modules.setup import init, add_newFAQs
from .modules.getFaq import findFAQ


def get_language(args):
    for key in args:
        if (type(args[key]) == bool and args[key] is True) or (args[key] is not None and type(args[key]) != bool):
            if key == "language":
                return args[key][0]
    return "it"


def check(parser, args):
    for key in args:
        if (type(args[key]) == bool and args[key] is True) or (args[key] is not None and type(args[key]) != bool):
            if key == "find":
                id = args[key][0]
                question = args[key][1]
                findFAQ(id, user_question=question, language=get_language(args), training=False)
            elif key == "setup":
                init()
            elif key == "train":
                id = args[key][0]
                file = args[key][1]
                add_newFAQs(id, file)
                findFAQ(id, language=get_language(args), training=True)
            return
    parser.print_help()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--setup", required=False, action='store_true', help="Run when using this framework for the first time.")
    parser.add_argument("-t", "--train", required=False, nargs=2, metavar=("<id>", "<file>"), help="Add the FAQs related to the <id> item and contained in <file>; each question has to be followed by its answer; every question/answer couple has to be separated from the others through a blank line.")
    parser.add_argument("-f", "--find", required=False, nargs=2, metavar=("<id>", "<question>"), help="Search <question> among the FAQs of the <id> item.")
    parser.add_argument("-l", "--language", required=False, nargs=1, metavar=("<language>"), help="Select the language to use among the following: it for Italian (default value) - en for English.")
    args = vars(parser.parse_args(argv))
    check(parser, args)


if __name__ == "__main__":
    main(sys.argv[1:])


# def main(argv):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-s", "--setup", required=False, action='store_true', help="Da lanciare soltanto la prima volta che si usa la libreria; inizializza le principali componenti.")
#     parser.add_argument("-t", "--train", required=False, nargs=2, metavar=("<id>", "<file>"), help="Aggiunge al sistema le FAQ relative al dispositivo identificato con <id> e contenute in <file>; ogni domanda deve essere direttamente seguita dalla risposta associata e ogni coppia domanda/risposta separata da un'altra tramite una riga vuota.")
#     parser.add_argument("-f", "--find", required=False, nargs=2, metavar=("<id>", "<question>"), help="Ricerca la domanda <question> tra le FAQ relative al dispositivo identificato tramite <id>.")
#     parser.add_argument("-l", "--language", required=False, nargs=1, metavar=("<language>"), help="Seleziona la lingua da usare tra le seguenti: it per italiano - en per inglese.")
#     args = vars(parser.parse_args(argv))
#     check(parser, args)
