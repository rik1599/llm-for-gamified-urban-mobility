# %%
import sys
sys.path.append('../../')

import warnings
warnings.filterwarnings('ignore')

from src.utils import metrics_scores
from tqdm.notebook import tqdm
import pandas as pd
from transformers import set_seed

set_seed(42)

tqdm.pandas()

# %%
from datasets import Dataset
import torch

prefix = "Fill <M> with either COMPLETED or FAILED: " 
size = 512

def preprocess(examples, tokenizer):
    inputs = [prefix + text for text in examples['X']]
    model_inputs = tokenizer(inputs, max_length=size, truncation=True)
    labels = tokenizer(examples['y'], max_length=8, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def compute_difficulty_scores(scores, tokens_ids):
    softmaxed_scores = [torch.nn.functional.softmax(t, dim=1) for t in scores]
    filtered_scores = [batch[:, token_id] for token_id, batch in zip(tokens_ids, softmaxed_scores)]
    filtered_scores = torch.column_stack(filtered_scores)
    return filtered_scores.mean(dim=1)

def predict(X, tokenizer, model):
    completed_tokens = list(tokenizer("COMPLETED").input_ids)
    failed_tokens = list(tokenizer("FAILED").input_ids)

    def batch_predicts(examples: Dataset):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=size, truncation=True, padding=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**model_inputs, max_new_tokens=4, do_sample=False, return_dict_in_generate=True, output_scores=True)
        output_str = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        return {
            "y": output_str,
            "failed_score": compute_difficulty_scores(outputs.scores, failed_tokens),
            "completed_score": compute_difficulty_scores(outputs.scores, completed_tokens)
        }
    
    ds = Dataset.from_dict({'text': X})
    ds = ds.map(batch_predicts, batched=True, batch_size=16)
    return (
        ds["y"], 
        ds["failed_score"], 
        ds["completed_score"]
    )

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

def cross_validation(df: pd.DataFrame, f, model_name='google-t5/t5-small', size=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=size)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args_ = Seq2SeqTrainingArguments(
        output_dir="tmp",
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=3,
        predict_with_generate=True,
        no_cuda=False,
        per_device_train_batch_size=4 if 'small' in model_name else 1,
        optim="adamw_torch",
    )

    weeks = df['cut'].unique()
    weeks = sorted(weeks)
    metrics = []
    inference = pd.DataFrame()

    for w in tqdm(weeks):
        train = df[(df['cut'] >= w - pd.Timedelta(weeks=4)) & (df['cut'] < w)][['X', 'y']]

        if train.empty:
            continue
        
        test = df[df['cut'] == w][['X', 'y']]

        train_ds = Dataset.from_pandas(train).map(
            lambda x: preprocess(x, tokenizer), batched=True)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args_,
            data_collator=data_collator,
            train_dataset=train_ds,
        )

        trainer.train()
        predict_df = predict(test['X'], tokenizer, model)
        test['y_pred'] = predict_df[0]
        test['failed_score'] = predict_df[1]
        test['completed_score'] = predict_df[2]
        test = test[test['y_pred'].isin(['COMPLETED', 'FAILED'])]
        
        metric = metrics_scores(test['y'], test['y_pred'], pos_label='COMPLETED')
        inference = pd.concat([inference, test])
       
        metric['week'] = w
        metrics.append(metric)
        print(metric)

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(f"../../out/csv/t5_metrics_{f}.csv", index=False)
    inference.to_parquet(f"../../out/parquet/t5_inference_{f}.parquet", index=True)

    model.save_pretrained(f'../../models/small/t5_{f}')
    tokenizer.save_pretrained(f'../../models/small/t5_{f}')


for f in [2, 3, 4, 5]:
    df = pd.read_parquet(f"../../out/parquet/prompts_{f}.parquet")
    cross_validation(df, f, model_name='google-t5/t5-small', size=512)


