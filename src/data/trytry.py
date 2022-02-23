import jsonlines

data = []
with jsonlines.open('/root/sparqling-queries/data/break/logical-forms-fixed/train_data_df.jsonl') as f:
    for d in f:
        data.append(
            {
                "input": d["input"],
                "output": d["output"],
            })

print(data[0]['output'])
