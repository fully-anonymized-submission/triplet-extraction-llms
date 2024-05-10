## Extracting semantic entity triplets by leveraging LLMs
This github repository provides the code that belongs to the paper "Extracting semantic entity triplets by leveraging LLMs". The code provides the backend for generating triplets with pre-trained LLMs. Furthermore, it allows the user to perform parameter-efficient fine-tuning (PEFT) before generating the triplets.

### Structure :books:
-- **src** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```configs_llm_extraction.py```: The settings for the triplet extraction \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```triplet_extraction_llm.py```: This file provides the code for the triplet extraction \
\
-- **data** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```train_data.csv```: The manually annotated train data \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```train_data.csv```: The manually annotated test data \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```fewshotsamples.json```: The few shot examples used for the triplet extraction \

\
-- ```requirements.txt```: File with environment requirements

### Environment :mag:
The file ```requirements.txt``` contains the requirements needed, which are compatible with python 3.11.7. Using the following code snippet, an environment can be created:

```
conda create -name <env_name> python=3.11.7 
pip install -r requirements.txt
```

### Data :page_facing_up:
The manually annotated training and test data originates from the paper [_"A Survey of Large Language Models"_](https://arxiv.org/abs/2303.18223).

### Parameter settings :pencil2:
For the parameter-efficient fine-tuning (PEFT), the following parameter settings are used:

| **Parameter**               | **Value** |
|-----------------------------|-----------|
| LoRa alpha                  | 64        |
| LoRa dropout                | 0.1       |
| LoRa r                      | 32        |
| Learning rate               | 0.0003    |
| Gradient accumulation steps | 4         |
| Weight decay                | 0.001     |
