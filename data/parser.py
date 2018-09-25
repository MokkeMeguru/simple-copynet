import MeCab as mecab
import re

""" 日本語の解析のためのユーティリティ """
ws = re.compile(' ')
wakati_tagger = mecab.Tagger('-Owakati')
hiragana_tagger = mecab.Tagger('-Oyomi')


def get_word_list(sentence):
    """
    Notice: returned list will not contain '、'.
    ['今日', 'は', '良い', '天気', 'です', 'ね', '。']
    :param sentence: simple text
    :return: list of word
    """
    return [x for x in ws.split(wakati_tagger.parse(sentence)) if x != '、'][:-1]


def get_hiragana_list(sentence):
    """
    Notice: returned list will not contain '、'.
    get_word_yomi_list('今日は、良い天気ですね。')
    ['キ', 'ョ', 'ウ', 'ハ', 'ヨ', 'イ', 'テ', 'ン', 'キ', 'デ', 'ス', 'ネ', '。']
    :param sentence: simple text
    :return: list of hiragana
    """
    return [x for x in hiragana_tagger.parse(sentence) if x != '、'][:-1]


def get_word_yomi_list(sentence):
    """
    Notice: returned list will not contain '、'.
    Example: get_word_yomi_list('今日は、良い天気ですね。')
    ['キョウ', 'ハ', 'ヨイ', 'テンキ', 'デス', 'ネ', '。']
    :param sentence: simple text
    :return: list of HIRAGANA word
    """
    return [hiragana_tagger.parse(x).rstrip()
            for x in ws.split(wakati_tagger.parse(sentence)) if x != '、'][:-1]


if __name__ == '__main__':
    sentence = '今日は、良い天気ですね。'
    print(sentence)
    print(get_word_list(sentence))
    print(get_word_yomi_list(sentence))
    print(get_hiragana_list(sentence))
