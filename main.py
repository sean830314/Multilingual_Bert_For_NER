from enum import Enum

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import wandb
from config import get_args
from metrics import BertBaseMultilingualMetrics

load_dotenv()


class Language(Enum):
    Korean = "ko"
    Japanese = "ja"


class Tokenizer(Enum):
    XLM_Roberta = "xlm-roberta-base"
    Multilingual_Bert = "bert-base-multilingual-cased"


class Preprocesser:
    def __init__(self, tokenizer: str, language: str):
        self.tokenizer = tokenizer
        self.language = language

    def get_tokenize_and_align_labels_func(self):
        if (
            self.tokenizer == Tokenizer.XLM_Roberta.value
            and self.language == Language.Japanese.value
        ):
            return self._japanese_xlm_rorberta_tokenize_and_align_labels
        else:
            return self._multilingual_tokenize_and_align_labels

    def _japanese_xlm_rorberta_tokenize_and_align_labels(self, all_samples_per_split):
        """
        use japanese text process
        """
        xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        text = ["".join(t) for t in all_samples_per_split["tokens"]]
        # tokenized_inputs = xlmr_tokenizer(text, truncation=True, max_length=512)
        tokenized_inputs = xlmr_tokenizer(text, truncation=True)

        #
        # map label to the new token
        #
        # [example]
        #   org token (all_samples_per_split)            : ["松", "崎", "は", "日", "本", "に", "い", "る"]
        #   new token (tokenized_inputs): ["_", "松", "崎", "は", "日本", "に", "いる"]
        labels = []
        for row_idx, label_old in enumerate(all_samples_per_split["ner_tags"]):
            # label is initialized as [[], [], [], [], [], [], []]
            label_new = [[] for t in tokenized_inputs.tokens(batch_index=row_idx)]
            # label becomes [[1], [1], [1], [0], [5, 5], [0], [0, 0]]
            for char_idx in range(len(all_samples_per_split["tokens"][row_idx])):
                token_idx = tokenized_inputs.char_to_token(row_idx, char_idx)
                if token_idx is not None:
                    label_new[token_idx].append(
                        all_samples_per_split["ner_tags"][row_idx][char_idx]
                    )
                    if (
                        tokenized_inputs.tokens(batch_index=row_idx)[token_idx] == "▁"
                    ) and (all_samples_per_split["ner_tags"][row_idx][char_idx] != 0):
                        label_new[token_idx + 1].append(
                            all_samples_per_split["ner_tags"][row_idx][char_idx]
                        )
            # label becomes [1, 1, 1, 0, 5, 0, 0]
            label_new = list(map(lambda i: max(i, default=0), label_new))
            # append result
            labels.append(label_new)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize all texts and align the labels with them.
    def _multilingual_tokenize_and_align_labels(self, all_samples_per_split):
        """
        use korean text process
        """
        total_adjusted_labels = []
        text_column_name = "tokens"
        label_column_name = "ner_tags"
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        tokenized_samples = tokenizer(
            all_samples_per_split[text_column_name],
            truncation=True,
            padding="max_length",
            max_length=512,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split[label_column_name][k]
            i = -1
            adjusted_label_ids = []

            for wid in word_ids_list:
                if wid is None:
                    adjusted_label_ids.append(-100)
                elif wid != prev_wid:
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    # label_name = label_names[existing_label_ids[i]]
                    adjusted_label_ids.append(existing_label_ids[i])
            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples["labels"] = total_adjusted_labels

        return tokenized_samples


def main():
    cfg = get_args()
    dataset = load_dataset("wikiann", cfg.language)
    label_names = dataset["train"].features["ner_tags"].feature.names
    id2label = {k: v for k, v in enumerate(label_names)}
    label2id = {v: k for k, v in enumerate(label_names)}
    print("-------------------- config --------------------")
    print(cfg)
    tokenize_and_align_labels = Preprocesser(
        cfg.tokenizer, cfg.language
    ).get_tokenize_and_align_labels_func()
    print(
        f"-------------------- tokenize_and_align_labels method is {tokenize_and_align_labels} --------------------"
    )
    print("-------------------- raw datasets --------------------")
    print(dataset)
    train_tokenized_dataset = dataset["train"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    validation_tokenized_dataset = dataset["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    test_tokenized_dataset = dataset["test"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["test"].column_names,
    )
    print("-------------------- train preprocess datasets --------------------")
    print(train_tokenized_dataset)
    print("-------------------- validation preprocess datasets --------------------")
    print(validation_tokenized_dataset)
    print("-------------------- test preprocess datasets --------------------")
    print(test_tokenized_dataset)
    if cfg.do_train:
        tokenizer = AutoTokenizer.from_pretrained(cfg.use_model)

        data_collator = DataCollatorForTokenClassification(tokenizer)

        model = AutoModelForTokenClassification.from_pretrained(
            cfg.use_model, id2label=id2label, label2id=label2id
        )
        training_args = TrainingArguments(
            output_dir=cfg.output_model_dir,
            max_steps=cfg.steps,
            save_steps=200,
            save_strategy="steps",
            # num_train_epochs=7,
            # weight_decay=0.01,
            # logging_steps = 1000,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=cfg.learning_rate,
            evaluation_strategy="steps",
            eval_steps=50,
            load_best_model_at_end=True,
            save_total_limit=2,
            metric_for_best_model="eval_overall_f1",
            report_to="wandb",
            run_name=f"language_{cfg.language}-{cfg.output_model_dir}-steps{cfg.steps}-lr{cfg.learning_rate}",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized_dataset,
            eval_dataset=validation_tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=BertBaseMultilingualMetrics(label_names).compute_metrics,
        )
        train_result = trainer.train()
        trainer.save_model(cfg.output_model_dir)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_metrics("eval", trainer.evaluate())
        wandb.finish()
    if cfg.do_eval:
        model = AutoModelForTokenClassification.from_pretrained(cfg.output_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(cfg.output_model_dir)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            args=TrainingArguments(
                output_dir=cfg.output_model_dir,
                report_to="wandb",  # enable logging to W&B
                run_name="eval-validation-dataset-model-{}-tokenizer-{}".format(
                    cfg.output_model_dir, cfg.tokenizer
                ),
            ),
            model=model,
            data_collator=data_collator,
            eval_dataset=validation_tokenized_dataset,
            compute_metrics=BertBaseMultilingualMetrics(label_names).compute_metrics,
        )
        print(trainer.evaluate())
        wandb.finish()
        trainer = Trainer(
            args=TrainingArguments(
                output_dir=cfg.output_model_dir,
                report_to="wandb",  # enable logging to W&B
                run_name="eval-test-dataset-{}-tokenizer-{}".format(
                    cfg.output_model_dir, cfg.tokenizer
                ),
            ),
            model=model,
            data_collator=data_collator,
            eval_dataset=test_tokenized_dataset,
            compute_metrics=BertBaseMultilingualMetrics(label_names).compute_metrics,
        )
        print(trainer.evaluate())
        wandb.finish()
    if cfg.do_predict:
        japan_t1 = [
            "#",
            "ヌ",
            "ン",
            "チ",
            "ャ",
            "ク",
            "バ",
            "ン",
            "キ",
            "：",
            "吉",
            "水",
            "孝",
            "宏",
        ]
        japan_t2 = [
            ":",
            "#",
            "鴨",
            "池",
            "新",
            "町",
            "。",
            "桜",
            "島",
            "・",
            "鹿",
            "児",
            "島",
            "湾",
            "・",
            "大",
            "隅",
            "半",
            "島",
            "が",
            "一",
            "望",
            "で",
            "き",
            "る",
            "。",
            "冬",
            "場",
            "の",
            "快",
            "晴",
            "で",
            "空",
            "気",
            "が",
            "澄",
            "ん",
            "で",
            "い",
            "る",
            "日",
            "に",
            "は",
            "遠",
            "く",
            "霧",
            "島",
            "山",
            "系",
            "や",
            "開",
            "聞",
            "岳",
            "も",
            "望",
            "め",
            "る",
            "。",
        ]
        japan_t3 = [
            "'",
            "'",
            "'",
            "ノ",
            "ー",
            "ネ",
            "'",
            "'",
            "'",
            "(",
            "'",
            "'",
            "'",
            "N",
            "o",
            "n",
            "e",
            "'",
            "'",
            "'",
            ")",
            "は",
            "、",
            "人",
            "口",
            "7",
            ",",
            "8",
            "6",
            "6",
            "人",
            "の",
            "イ",
            "タ",
            "リ",
            "ア",
            "共",
            "和",
            "国",
            "ピ",
            "エ",
            "モ",
            "ン",
            "テ",
            "州",
            "ト",
            "リ",
            "ノ",
            "県",
            "の",
            "コ",
            "ム",
            "ー",
            "ネ",
            "の",
            "一",
            "つ",
            "で",
            "あ",
            "る",
            "。",
        ]
        japan_text = (
            "".join(japan_t1) + "," + "".join(japan_t2) + "," + "".join(japan_t3)
        )
        korea_text = (
            "현재 대한민국 K리그 챌린지의 서울 이랜드 FC에서 활약하고 있다 ."
            + " "
            + " ".join(
                [
                    "브루나이",
                    ",",
                    "중화민국",
                    ",",
                    "타이",
                    ",",
                    "중화인민공화국",
                    "(",
                    "홍콩",
                    "포함",
                    ")",
                    ",",
                    "일본",
                    ",",
                    "싱가포르",
                    ",",
                    "말레이시아",
                    ",",
                    "인도네시아",
                    ",",
                    "캄보디아",
                    "등에도",
                    "수출되었다",
                    ".",
                ]
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.output_model_dir)
        model = AutoModelForTokenClassification.from_pretrained(cfg.output_model_dir)
        from transformers import pipeline

        if cfg.language == "ko":
            # print("---------- custom output ----------")
            # evaluate_one_text(tokenizer, model, korea_text)
            print("---------- transformers pipeline output ----------")
            recognizer = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="average",
            )  # korea
            print(recognizer(korea_text))
        elif cfg.language == "ja":
            print("---------- transformers pipeline output ----------")
            recognizer = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )  # japan
            print(recognizer(japan_text))


def align_word_ids(word_ids):
    label_all_tokens = False
    # tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    # word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            label_ids.append(1)
        else:
            label_ids.append(1 if label_all_tokens else -100)
        previous_word_idx = word_idx

    return label_ids


def get_tokens(tokenizer, word_ids, tokens_ids):
    tokens = []
    temp_list = []
    pre_id = -1
    for index, id in enumerate(word_ids):
        if id is not None:
            if id == pre_id:
                temp_list[-1] = (temp_list[-1][0], temp_list[-1][1] + 1)
            else:
                pre_id = id
                temp_list.append((index, index + 1))
    for item in temp_list:
        tokens.append(tokenizer.decode(tokens_ids[item[0] : item[1]]))

    return tokens


def convert_bio_to_entities(paragraph, tokens, predictions):
    entities = list()
    previous_label = "O"
    previous_index = -1
    temp_entities = list()
    for index, token, pred in zip(range(0, len(tokens)), tokens, predictions):
        if pred == "O":
            previous_label = "O"
            previous_index += 1
            continue
        else:
            if pred[0:2] == "B-":
                temp_ent = {
                    "text": "",
                    "label": pred[2:],
                    "words": [
                        {
                            "text": token.lstrip().rstrip(),
                        }
                    ],
                }
                temp_entities.append(temp_ent)
                previous_label = pred[2:]
            elif (
                pred[0:2] == "I-"
                and len(temp_entities) != 0
                and temp_entities
                and previous_label == pred[2:]
                and previous_index == index - 1
            ):
                temp_entities[-1]["words"].append({"text": token.lstrip().rstrip()})
            previous_index += 1
    for temp_entity in temp_entities:
        temp_entity["text"] = " ".join(word["text"] for word in temp_entity["words"])
        index = paragraph.find(temp_entity["text"])
        temp_entity["offset"] = index if index != -1 else -1
        temp_entity["length"] = len(temp_entity["text"])
        entities.append(temp_entity)
    return {"paragraph": paragraph, "entities": entities}


def evaluate_one_text(tokenizer, model, sentence):
    """
    korea inference code by original method
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    text = tokenizer(
        sentence,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    mask = text["attention_mask"].to(device)
    input_id = text["input_ids"].to(device)

    # get array of align words
    label_ids = torch.Tensor(align_word_ids(text.word_ids())).unsqueeze(0).to(device)

    # get true predictions
    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]
    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [model.config.id2label[p] for p in predictions]

    # get true tokens
    tokens = get_tokens(tokenizer, text.word_ids(), text["input_ids"][0].tolist())

    assert len(tokens), len(predictions)
    for token, prediction in zip(tokens, prediction_label):
        print(token, "\t", prediction)
    result = convert_bio_to_entities(sentence, tokens, prediction_label)
    print(result)


if __name__ == "__main__":
    main()
