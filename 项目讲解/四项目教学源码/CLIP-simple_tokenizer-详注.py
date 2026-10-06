# 教学副本：与 CLIP/clip/simple_tokenizer.py 保持可执行结构一致，只增加中文解释。
# 阅读顺序：先看 import 与 encode；需要弄懂 BPE 合并时再看 bpe()。
# gzip 读取被压缩成 .gz 的 BPE 合并规则文件。
import gzip
# html.unescape 将 &amp; 一类 HTML 实体恢复为普通字符。
import html
# os.path 用于定位与本脚本位于同一目录的词表文件。
import os
# lru_cache 缓存函数结果；第二次调用无参函数不必重复构造同一映射。
from functools import lru_cache

# ftfy 修复常见乱码/排版问题；输入输出都是文字字符串。
import ftfy
# regex 支持 Unicode 字母类别；别名 re，和标准库 re 名字相同但不是同一包。
import regex as re


@lru_cache()
def default_bpe():
    # __file__ 是当前 Python 文件路径；abspath 转绝对路径，dirname 取文件所在目录。
    # join 再拼出压缩 BPE 词表文件路径。返回的是字符串，不是文件内容。
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    # bs=bytes，先选已有可打印字符对应的整数编码。ord('!') 得到该字符编号。
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    # cs=characters 的编码列表；[:] 复制，后面增加新字符时不改 bs。
    cs = bs[:]
    # n 数未覆盖的字节，给它们分配 256 以上的新字符编号。
    n = 0
    # 2**8=256，遍历单字节能表示的 0..255 全部整数。
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    # chr(整数) 变 Unicode 字符；zip 把每个字节与对应字符配成一对。
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # word 是当前 BPE 片段元组；set 去重保存所有相邻片段对。
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        # 例如 ('c','a','t') 得 ('c','a') 与 ('a','t') 两对。
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    # 从内到外：先修乱码，再两次 HTML 实体解码，最后去两端空白。
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    # 正则 \s+ 匹配连续空白；re.sub 用单个普通空格替换。
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        # bpe_path 是 .gz 合并规则文件路径；: str 是类型提示，不会自动打开文件。
        # 先把任意 UTF-8 字节映射到可打印字符，再用 BPE 合并常见字符组合。
        self.byte_encoder = bytes_to_unicode()
        # 字典推导式交换键和值，供 decode 将可打印字符还原成原字节。
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # merges 文件记录“哪两个片段优先合并”，顺序本身就是优先级。
        # gzip.open 读压缩字节，decode('utf-8') 转字符串，split('\n') 拆各行。
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        # 跳过文件头，只取 CLIP 词表规定数量的合并规则；切片上界不包含自身。
        merges = merges[1:49152-256-2+1]
        # 每行如两个片段 a b；split 拆开、tuple 变成可做字典键的二元组。
        merges = [tuple(merge.split()) for merge in merges]
        # vocab=vocabulary（词表）；先放所有单字节可表示的字符。
        vocab = list(bytes_to_unicode().values())
        # </w> 是 word end（词尾）标记；词中字符与词尾字符视为不同 token。
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            # 两段字符串直接拼接；常见组合因此成为词表中的一个新 token。
            vocab.append(''.join(merge))
        # SOT=start of text，EOT=end of text。它们最后加入，EOT ID 因而很大。
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        # encoder 是“子词字符串 → 整数 ID”的字典；神经网络收到整数而非字符串。
        self.encoder = dict(zip(vocab, range(len(vocab))))
        # decoder 是反向“整数 ID → 子词字符串”的字典。
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 数字越小代表合并优先级越高；bpe() 会反复选排名最靠前的相邻对。
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # cache 缓存已切过的 token；两个特殊符号直接映到自己。
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        # regex 模式把特殊词、缩写、Unicode 字母、数字与标点拆开。
        # re.IGNORECASE 让正则匹配不区分英文大小写。
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        # 缓存已切过的词，避免同一个词在许多句子中重复运行 BPE。
        if token in self.cache:
            return self.cache[token]
        # 将 token 拆为逐字符片段，最后片段加词尾标记；(x,) 是单元素元组。
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        # pairs 是当前片段序列中所有相邻二元组，供下一步选择合并。
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        # 每次挑一对排名最高的相邻片段合并，直到找不到已知规则。
        while True:
            # 从当前所有相邻片段对中选择训练词表里最优先的那一对。
            # lambda 读作“给一对 pair，返回它的排名”；未知 pair 排名是无穷大。
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            # 元组拆包：本轮要合并的左片段 first 与右片段 second。
            first, second = bigram
            # new_word 临时收集本轮扫描/合并后的片段，原 word 是元组不能原地改。
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # 从 i 往后寻找 next first；index 找不到会抛 ValueError。
                    j = word.index(first, i)
                    # i..j 之前未命中，原样追加到新列表。
                    new_word.extend(word[i:j])
                    i = j
                except:
                    # 原仓库用宽泛 except；这里通常处理找不到 first 的情形。
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    # 真正合并两个相邻片段，下一轮再重新检查相邻对。
                    new_word.append(first+second)
                    # 两个旧片段已消耗，索引向右走两格。
                    i += 2
                else:
                    # 未恰好出现目标二元组时保留当前片段，只走一格。
                    new_word.append(word[i])
                    i += 1
            # 本轮结果覆盖局部变量 word；这里 word 已不是原始逐字符片段。
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                # 新片段边界改变了，重新计算相邻对才能进入下一轮。
                pairs = get_pairs(word)
        # BPE 的结果暂用空格分隔；encode() 会按空格拆开查词表 ID。
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        # text 是原始 Python 句子字符串；bpe_tokens 最终存整数 ID 列表。
        bpe_tokens = []
        # 从内到外清乱码/HTML、合并空白、转英文小写。
        text = whitespace_clean(basic_clean(text)).lower()
        # 正则先拆出词、数字、标点，再对每段执行 byte-level BPE。
        for token in re.findall(self.pat, text):
            # token.encode('utf-8') 得字节；逐字节查可打印字符，再拼回字符串。
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # BPE 片段转成整数 ID；模型收到的是 ID，不是 Python 字符串。
            # split(' ') 把 bpe() 的空格分隔片段拆开；extend 将所有 ID 接到整句后面。
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        # 此处还没有加 SOT/EOT，也未补齐 77 位；外层 clip.tokenize() 才做。
        return bpe_tokens

    def decode(self, tokens):
        # tokens 是整数 ID 序列；先反查各子词，再把字节映射字符还原成 UTF-8。
        text = ''.join([self.decoder[token] for token in tokens])
        # byte_decoder[c] 把字符变原字节；bytearray 汇成字节流；
        # errors='replace' 遇到不合法序列用替代字符；最后将词尾 </w> 改为空格。
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
