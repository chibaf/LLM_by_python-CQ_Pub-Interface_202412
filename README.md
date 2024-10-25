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
## references

csebuetnlp/xl-sum: This repository contains the code, data, and models of the paper titled "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages" published in Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021.

https://github.com/csebuetnlp/xl-sum?tab=readme-ov-file#datasets

12月号　Pythonで動かして学ぶ線形代数[LLM/姿勢推定/信号処理/GPS] 　

https://www.cqpub.co.jp/interface/download/contents2024.htm
