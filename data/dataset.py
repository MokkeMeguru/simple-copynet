import random

import torch
from torch.utils.data import Dataset
from operator import itemgetter


class Language(object):
    def __init__(self, vocab_limit, data_list, remove_words=[]):
        self.data_list = data_list
        self.remove_words = remove_words
        self.vocab = self.create_vocab()
        print('vocab_limit', vocab_limit)
        truncated_vocab = sorted(self.vocab.items(), key=itemgetter(1), reverse=True)[:vocab_limit]
        self.tok_to_idx = dict()
        self.tok_to_idx['<MSK>'] = 0
        self.tok_to_idx['<SOS>'] = 1
        self.tok_to_idx['<EOS>'] = 2
        self.tok_to_idx['<UNK>'] = 3
        # add some important keywords
        # self.tok_to_idx['。'] = 4
        # self.tok_to_idx['、'] = 5
        for idx, (tok, _) in enumerate(truncated_vocab):
            self.tok_to_idx[tok] = idx + 4
        self.idx_to_tok = {idx: tok for tok, idx in self.tok_to_idx.items()}

    def create_vocab(self):
        vocab = dict()
        for data in self.data_list:
            for sentence in data:
                for token in sentence:
                    if (not token == '') and (token not in self.remove_words):
                        vocab[token] = vocab.get(token, 0) + 1
        return vocab


class SequencePairDataset(Dataset):
    def __init__(self,
                 data_list,
                 maxlen=30,
                 lang=None,
                 vocab_limit=None,
                 val_size=0.1,
                 seed=42,
                 is_val=False,
                 use_cuda=False,
                 use_extended_vocab=True,
                 remove_words=[]):
        self.input_seqs = data_list[0]
        self.output_seqs = data_list[1]
        self.maxlen = maxlen
        self.use_cuda = use_cuda
        self.parser = None
        self.val_size = val_size
        self.seed = seed
        self.is_val = is_val
        self.use_extended_vocab = use_extended_vocab

        idxs = list(range(len(self.input_seqs)))
        random.seed(self.seed)
        random.shuffle(idxs)
        num_val = int(len(idxs) * self.val_size)
        if self.is_val:
            self.idxs = idxs[:num_val]
        else:
            self.idxs = idxs[num_val:]
        self.input_seqs = [self.input_seqs[idx] for idx in idxs]
        self.output_seqs = [self.output_seqs[idx] for idx in idxs]
        if lang is None:
            lang = Language(vocab_limit, data_list, remove_words)
        self.lang = lang

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        input_token_list = (['<SOS>'] + self.input_seqs[idx] + ['<EOS>'])[:self.maxlen]
        output_token_list = (['<SOS>'] + self.output_seqs[idx] + ['<EOS>'])[:self.maxlen]

        input_seq = self.tokens_to_seq(input_token_list)
        output_seq = self.tokens_to_seq(output_token_list, input_token_list=input_token_list)
        if self.use_cuda:
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()
        return input_seq, output_seq, ''.join(input_token_list), ''.join(output_token_list)

    def tokens_to_seq(self, token_list, input_token_list=None):
        seq = torch.zeros(self.maxlen).long()
        tok_to_idx_extension = dict()
        for pos, token in enumerate(token_list):
            if token in self.lang.tok_to_idx:
                # token が語彙中に存在していた場合
                idx = self.lang.tok_to_idx[token]
            elif token in tok_to_idx_extension:
                # token が 拡張された語彙中に存在している場合
                idx = tok_to_idx_extension[token]
            elif self.use_extended_vocab and input_token_list is not None:
                # If the token is not in the vocab and an input token sequence was provided
                #  find the position of the first occurrence of the token in the input sequence
                #  the token index in the output sequence is size of the vocab plus the position in the input sequence.
                # If the token cannot be found in the input sequence use the unknown token.
                # もしその token が入力シーケンスにあって、語彙中に存在しない場合、
                # その入力シーケンスの token が一番始めに出現した位置に語彙数を加算した値を割り当てます。
                # (※下の条件分岐を参照)
                # もし入力シーケンスにそれが存在しなければ <UNK> とします。
                # hint:
                # next((t + 100 for t, x in enumerate(['w', 'x', 'y', 'z']) if x == 'x'), 3) => 101
                # next((t + 100 for t, x in enumerate(['w', 'x', 'y', 'z']) if x == 'a'), 3) => 3 (= <UNK>)
                tok_to_idx_extension[token] = tok_to_idx_extension.get(token,
                                                                       next((pos + len(self.lang.tok_to_idx)
                                                                             for pos, input_token
                                                                             in enumerate(input_token_list)
                                                                             if input_token == token), 3))
                idx = tok_to_idx_extension[token]
            elif self.use_extended_vocab:
                # unknown tokens in the input sequence use
                # the position of the first occurrence + vocab_size as their index
                # 入力シーケンス中に表れた未知語は、最初に表れた場所に語彙数を加算した値を割当てます。
                idx = pos + len(self.lang.tok_to_idx)
            else:
                idx = self.lang.tok_to_idx['<UNK>']
            seq[pos] = idx
        return seq

    def seq_to_string(self, seq, input_tokens=None):
        """

        :param seq:
        :param input_tokens:
        :return:
        """
        vocab_size = len(self.lang.idx_to_tok)
        seq_length = (seq != 0).sum()
        words = []
        for idx in seq[:seq_length]:
            # ???
            idx = int(idx.cpu().numpy())
            if idx < vocab_size:
                words.append(self.lang.idx_to_tok[idx])
            elif input_tokens is not None:
                words.append(input_tokens[idx - vocab_size])
            else:
                words.append('<???>')
        string = ''.join(words)
        return string


if __name__ == '__main__':
    from data import parser
    from data import reader
    data_list = reader.read_files('../st-data/base.csv', '../st-data/styled.csv', parser.get_word_list)
    spdataset = SequencePairDataset(data_list, vocab_limit=100)
    x = spdataset.__getitem__(1)
    print(x)
