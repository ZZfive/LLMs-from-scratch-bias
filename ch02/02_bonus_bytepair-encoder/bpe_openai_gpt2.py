# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# License:
# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import regex as re
import requests
from tqdm import tqdm
from functools import lru_cache


"""
一些具体的映射例子：
1. ASCII可打印字符：
字节值97（'a'）→ Unicode字符'a'（码点97）
字节值65（'A'）→ Unicode字符'A'（码点65）
2. 控制字符：
字节值0（NUL）→ Unicode字符'Ā'（码点256）
字节值10（换行符）→ Unicode字符'Ċ'（码点266）
3. 特殊字符：
字节值32（空格）→ Unicode字符'Ġ'（码点288）
字节值9（制表符）→ Unicode字符'Ĉ'（码点264）
"""
@lru_cache()  # 使用lru_cache装饰器来缓存结果，避免重复计算
def bytes_to_unicode():  # 返回utf-8字节列表和相应的unicode字符串列表，可将任何utf-8字节序列转换为BPE可以处理的unicode字符串，同时避免特殊字符问题
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.

    返回utf-8字节列表和相应的unicode字符串列表。
    可逆的bpe编码适用于unicode字符串。
    这意味着，如果你想避免UNK（未知标记），你需要在词汇表中包含大量的unicode字符。
    当你处理约100亿token的数据集时，你最终需要大约5000个字符才能获得合理的覆盖率。
    这在你通常的、比如说32K的bpe词汇表中占据了相当大的比例。
    为了避免这种情况，我们需要在utf-8字节和unicode字符串之间建立查找表。
    并且避免映射到空白字符/控制字符，这些字符会导致bpe代码出错。
    """
    # 收集可打印字符的字节值，包含三组字符，ASCII可打印字符、拉丁-1补充字符的前半部分和后半部分
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):  # 遍历所有可能的字节值
        if b not in bs:  # 对于不在初始bs列表中的字节值（主要是控制字符和一些特殊字符）
            bs.append(b)  # 将其添加到bs列表中
            cs.append(2**8 + n)  # 为不在初始bs列表中的字节值重新分配一个唯一的Unicode编码，即从256开始
            n += 1
    cs = [chr(n) for n in cs]  # 将cs列表中的Unicode编码转换为对应的Unicode字符；可打印字符映射到它们原始的Unicode编码，控制字符和特殊字符映射到256以上的自定义Unicode编码
    return dict(zip(bs, cs))  # 返回一个字典，键为字节值，值为对应的Unicode字符


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

'''
re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")针对的匹配项
1. 's|'t|'re|'ve|'m|'ll|'d'：匹配英语常见缩写后缀
    's：所有格或缩写形式，如"it's"、"John's"
    't：not的缩写，如"don't"、"won't"
    're：are的缩写，如"you're"、"they're"
    've：have的缩写，如"I've"、"could've"
    'm：am的缩写，如"I'm"
    'll：will的缩写，如"I'll"、"she'll"
    'd：would/had的缩写，如"I'd"、"he'd"
2. | ?\p{L}+：匹配（可能带前导空格的）一个或多个字母字符
3. | ?\p{N}+：匹配（可能带前导空格的）一个或多个数字字符
4. | ?[^\s\p{L}\p{N}]+：匹配（可能带前导空格的）一个或多个非空白、非字母、非数字的字符
5. |\s+(?!\S)：匹配末尾的空白字符
6. |\s+：匹配任何空白字符序列
'''
class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder  # 词汇表中token到id的映射
        self.decoder = {v: k for k, v in self.encoder.items()}  # id到token的映射
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()  # 将utf-8字节转换为unicode字符串的查找表
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}  # 将unicode字符串转换为utf-8字节的查找表
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # 将bpe合并规则转换为字典，键为合并规则，值为合并规则的索引
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]  # 如果token已经在缓存中，直接返回缓存中的结果
        word = tuple(token)  # 将token转换为元组，会就一个字符串全部拆开，如Hello-->('H', 'e', 'l', 'l', 'o')
        pairs = get_pairs(word)  # 获取所有可能的符号对

        if not pairs:
            return token

        # 不断合并优先级最高的符号对，直到没有可以合并的符号对为止
        while True:
            # 找出所有字符对中优先级最高的一对（在bpe_ranks中排名最低的）
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:  # 如果bigram不在bpe_ranks中，说明已经没有可以合并的符号对，退出循环
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)  # 从索引i开始向后搜索first出现的位置
                    new_word.extend(word[i:j])  # 将i到j之间的所有字符添加到new_word
                    i = j
                except ValueError:
                    new_word.extend(word[i:])  # 如果找不到first，将剩余所有字符添加到new_word并退出循环
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # 如果当前位置是first且下一个位置是second，则将它们合并为一个新的子词单元
                    i += 2
                else:
                    new_word.append(word[i])  # 否则，只添加当前字符
                    i += 1
            new_word = tuple(new_word)
            word = new_word  # 更新
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word  # 缓存结果
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):  # 使用正则表达式将文本分割成初始token
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))  # 将初始token转换为unicode字符串
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))  # 将初始token转换为bpe token
        return bpe_tokens
    
    def encode_bias(self, text):
        bpe_tokens = []
        tokens = re.findall(self.pat, text)
        for token in tokens:  # 使用正则表达式将文本分割成初始token
            token_tmp = ""
            for b in token.encode('utf-8'):
                tmp = self.byte_encoder[b]
                token_tmp += tmp
            tmp_bpe_tokens = self.bpe(token_tmp).split(' ')
            for bpe_token in tmp_bpe_tokens:
                bpe_tokens.append(self.encoder[bpe_token])
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text
    
    def decode_bias(self, tokens):
        subtexts = []
        for token in tokens:
            subtext = self.decoder[token]
            subtexts.append(subtext)
        text = "".join(subtexts)
        ids = []
        for c in text:
            ids.append(self.byte_decoder[c])
        text = bytearray(ids).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    # Modified code from
    subdir = 'gpt2_model'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for filename in ['encoder.json', 'vocab.bpe']:
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


if __name__ == "__main__":
    encoder = get_encoder('gpt2_model', '/root/code/LLMs-from-scratch-bias/ch02/02_bonus_bytepair-encoder')
    text = "Hello, world!"
    tokens = encoder.encode_bias(text)
    print(tokens)
    decoded_text = encoder.decode_bias(tokens)
    print(decoded_text)
