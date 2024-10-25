def read_data(file_path):
    # 学習データセットを保持するリスト
    import json
    texts = []
    summaries = []

    # JSONLファイルから text と summary を抽出
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            text = json_data.get("text")
            summary = json_data.get("summary")
            if text and summary:
                texts.append(text)
                summaries.append(summary)
    return texts, summaries
