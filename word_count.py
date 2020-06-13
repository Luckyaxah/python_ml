# coding = utf-8

__author__ = 'nowcoder'

import re


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    return False

def contain_chinese(ustr):
    """
    判断一个unicode字符串是否含有中文
    """
    for uchar in ustr:
        if is_chinese(uchar):
            return True
        else:
            return False

def strip_symbols(ustr):
    """
    删除英文标点符号
    """
    return re.sub(u'[,\[`!\"#$%&\'\(\)*\+-/:;<=>?@\]\\^_{|}~\.]',"",ustr )

def count_words(filename):
    import collections # 容器模块
    words_count = collections.Counter() # 内部是一个dict，会自动计数

    with open(filename, 'rb') as fp:
        for line in fp:
            line = line.decode('utf-8').strip()
            if contain_chinese(line):
                continue

            line = strip_symbols(line) # 删除语文标点符号
            print(line)
            line = line.lower()

            words_count.update(re.split(r'\s+', line))
    # 使用dict返回是因为words_count对于不存在的键会返回0
    return dict(words_count)


def get_top(filename, topk = 10):
    words_dict = count_words(filename)

    top_words = sorted(words_dict.items(), key=lambda x: x[1], reverse= True)
    return top_words[:topk]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: {} filename topk".format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)
    # argv都是string类型
    topwords = get_top(sys.argv[1], int(sys.argv[2]))


    for word, count in topwords:
        print(word, count)
