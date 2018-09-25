from multimethod import multimethod


@multimethod
def read_files(data_paths: list, parse_func):
    """
    read data from files
    files' format: csv
    -----------------------------
    input text, output text
    input text, output text
    -----------------------------
    notice:
     parse func: text -> list of word
    :param data_paths: list of data paths
    :param parse_func: parse function such as splitting with space
    :return: [[[input seq1][input seq2] ... ] [[output seq1] ... ]]
    """
    input_seqs = []
    output_seqs = []
    for data_path in data_paths:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line is not '':
                    input_seq, output_seq = line.split(',')[:2]
                    input_seqs.append(parse_func(input_seq))
                    output_seqs.append(parse_func(output_seq))
    return [input_seqs, output_seqs]


@multimethod
def read_files(base_file: str, styled_file: str, parse_func):
    """
    read data from two file.
    one contains input data, another contains output data
    :param base_file: input text data path
    :param styled_file: output text data path
    :param parse_func: parse function such as splitting with space
    :return: [[[input seq1][input seq2] ... ] [[output seq1] ...]]
    """
    return [[parse_func(seq) for seq in open(file_name, 'r', encoding='utf-8') if not seq == '']
            for file_name in [base_file, styled_file]]


if __name__ == '__main__':
    from data import parser
    input_seqs, output_seqs = read_files('../st-data/base.csv', '../st-data/styled.csv', parser.get_word_list)
    print('{}: {}'.format(len(input_seqs), len(input_seqs) == len(output_seqs)))
    print(input_seqs[:3])
    print(output_seqs[:3])
