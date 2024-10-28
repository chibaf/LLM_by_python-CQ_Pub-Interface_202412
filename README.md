# LLM_by_python-interface-202412

## 第３章 Pythonで作って学ぶLLM(大規模言語モデル）

## programs ran on Macbook Air M1

## installation of pytorch

$ pip install torch

if failed, $ pip3 install torch --break-system-packages

## installation of pytorch : Raspberry Pi5

pip3 install 'torch<2.5' --break-system-packages

## check 1

$ python3 check.py

if no error, check.py prints "check ended."

## check 2

<pre>
>>> import torch<br>
>>> import s_rd     #import reading data set subprogram=s_rd.py<br>
>>> file_path_train='japanese_train.jsonl'<br>
>>> file_path_val='japanese_val.jsonl'<br>
>>> texts,summaries=s_rd.read_data(file_path_train)<br>
>>> print(texts[0])   #全文<br>
救出作戦の間、洞窟内に少年たちと留まったタイ海軍のダイバーと医師も最後に無事脱出した。4人の写真は10日、タイ海軍特殊部隊がフェイスブックに掲載したもの タイ海軍特殊部隊はフェイスブックで、「これは奇跡なのか科学なのか、一体何なのかよくわからない。『イノシシ』13人は全員、洞窟から出た」と救助作戦の終了を報告した。「イノシシ」（タイ語で「ムーパ」）は少年たちの所属するサッカー・チームの愛称。 遠足に出かけた11歳から17歳の少年たちと25歳のサッカー・コーチは6月23日、大雨で増水した洞窟から出られなくなった。タイ内外から集まったダイバー約9....<br>
>>> print(summaries[0])  #要約<br>
タイ北部のタムルアン洞窟で10日夜、中に閉じ込められていた少年12人とサッカー・コーチの計13人のうち、最後の少年4人とコーチが水路を潜り無事脱出した。その約3時間後には、洞窟内で少年たちと留まっていた海軍ダイバー3人と医師も生還した。17日間も洞窟内にいた13人の救出に、タイ国内外で多くの人が安心し、喜んでいる。<br>
</pre>

## check 3
<pre>
python3 testjson.py <br>
all text:<br>
救出作戦の間、洞窟内に少年たちと留まったタイ海軍のダイバーと医師も最後に無事脱出した。4人の写真は10日、タイ海軍...<br>
（英語記事 Cave rescue: Elation as Thai boys and coach freed by divers）<br>
summary:<br>
タイ北部のタムルアン洞窟で10日夜、中に閉じ込められていた少年12人とサッカー・コーチの計13人のうち、最後の少年4人とコーチが水路を潜り無事脱出した。その約3時間後には、洞窟内で少年たちと留まっていた海軍ダイバー3人と医師も生還した。17日間も洞窟内にいた13人の救出に、タイ国内外で多くの人が安心し、喜んでいる。<br>
</pre>

## 3-4 実装(1)トークナイザ

<pre>
import json<br>
import torch<br>
import torch.nn as nn<br>
from torch.nn import functional as F<br>
<br>
# hyperparameters<br>
batch_size = 16 # how many independent sequences will we process in parallel?<br>
block_size = 500 # what is the maximum context length for predictions?<br>
max_iters = 50000000<br>
eval_interval = 100<br>
learning_rate = 1e-3<br>
device = 'cuda' if torch.cuda.is_available() else 'cpu'<br>
eval_iters = 200<br>
n_embd = 64<br>
n_head = 4<br>
n_layer = 4<br>
dropout = 0.0<br>
# ------------<br>
<br>
torch.manual_seed(1337)<br>
<br>
def read_data(file_path):<br>
    # 学習データセットを保持するリスト<br>
    texts = []<br>
    summaries = []<br>
<br>
    # JSONLファイルから text と summary を抽出<br>
    with open(file_path, 'r', encoding='utf-8') as file:<br>
        for line in file:<br>
            json_data = json.loads(line)<br>
            text = json_data.get("text")<br>
            summary = json_data.get("summary")<br>
            if text and summary:<br>
                texts.append(text)<br>
                summaries.append(summary)<br>
    return texts, summaries<br>
<br>
# Tokenizerクラスの実装<br>
class Tokenizer:<br>
<br>
    @staticmethod<br>
    def create_vocab(dataset):<br>
        """<br>
        Create a vocabulary from a dataset.<br>
<br>
        Args:<br>
            dataset (str): Text dataset to be used to create the character vocab.<br>
<br>
        Returns:<br>
            Dict[str, int]: Character vocabulary.<br>
        """<br>
        vocab = {<br>
            token: index<br>
            for index, token in enumerate(sorted(list(set(dataset))))<br>
        }<br>
<br>
        # Adding unknown token<br>
        vocab["<unk>"] = len(vocab)<br>
<br>
        return vocab<br>
<br>
    def __init__(self, vocab):<br>
        """<br>
        Initialize the tokenizer.<br>
<br>
        Args:<br>
            vocab (Dict[str, int]): Vocabulary.<br>
        """<br>
        self.vocab_encode = {str(k): int(v) for k, v in vocab.items()}<br>
        self.vocab_decode = {v: k for k, v in self.vocab_encode.items()}<br>
<br>
    def encode(self, text):<br>
        """<br>
        Encode a text in level character.<br>
<br>
        Args:<br>
            text (str): Input text to be encoded.<br>

        Returns:<br>
            List[int]: List with token indices.
        """
        return [self.vocab_encode.get(char, self.vocab_encode["<unk>"]) for char in text]<br>
<br>
    def decode(self, indices):<br>
        """<br>
        Decode a list of token indices.<br>
<br>
        Args:<br>
            indices (List[int]): List of token indices.<br>
<br>
        Returns:<br>
            str: The decoded text.<br>
        """<br>
        return "".join([self.vocab_decode.get(idx, "<unk>") for idx in indices])<br>
</pre>

### sample code
https://github.com/chibaf/LLM_by_python-CQ_Pub-Interface_202412/blob/main/tokenizer_test.py

## 実装（２）自動要約AIを作成するためのデータの準備

## references

csebuetnlp/xl-sum: This repository contains the code, data, and models of the paper titled "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages" published in Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021.

https://github.com/csebuetnlp/xl-sum?tab=readme-ov-file#datasets

12月号　Pythonで動かして学ぶ線形代数[LLM/姿勢推定/信号処理/GPS] 　

https://www.cqpub.co.jp/interface/download/contents2024.htm
