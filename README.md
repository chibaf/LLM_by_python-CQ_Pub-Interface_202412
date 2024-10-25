# MML_by_python-interface-202410
MML_by_python-interface-202410

## programs ran on Macbook Air M1

## installation of pytorch

$ pip install torch

if failed, $ pip3 install torch --break-system-packages

## check 1

$ python3 check.py

if noerror, check.py prints "check ended."

## check 2

<pre>
>>> import json<br>
>>> import s_rd<br>
>>> file_path_train='japanese_train.jsonl'<br>
>>> file_path_val='japanese_val.jsonl'<br>
>>> texts,summaries=s_rd.read_data(file_path_train)<br>
>>> print(texts[0])<br>

</pre>

## references

csebuetnlp/xl-sum: This repository contains the code, data, and models of the paper titled "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages" published in Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021.

https://github.com/csebuetnlp/xl-sum?tab=readme-ov-file#datasets
