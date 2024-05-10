import pandas as pd
from datasets import Dataset
import numpy as np
from helpers import get_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, PeftModel
import torch
from pathlib import Path
from trl import SFTTrainer
import os
import time
import json
import regex
from accelerate import PartialState
from transformers import BitsAndBytesConfig
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import configs_llm_extraction
from estimation import find_total_usable_memory, estimate_memory

from transformers import set_seed
set_seed(0)

def get_prompt_string_starling(raw_text, tokenizer, output=None, use_fewshot_prompting=True, few_shot_examples=None, summarize=False):
    """
    Create a prompt string for the Starling model - this requires a separate function because Starling does not support the chat template

    Args:
    raw_text: str: The text to be used in the prompt
    tokenizer: AutoTokenizer: The tokenizer to be used
    output: str: The output to be used in the prompt
    use_fewshot_prompting: bool: Whether to use fewshot prompting
    few_shot_examples: list: The few shot examples to be used
    summarize: bool: Whether to summarize the text

    Returns:
    str: The prompt string
    """
    if use_fewshot_prompting:
        chat = few_shot_examples.copy()
        chat.append({"role": "user", "content": "The next text is: \n {}".format(raw_text)})
    else:
        # throw error asking whether you truly do not want to use fewshot prompting
        if summarize:
            chat = [{"role": "user", "content": "Make a summary of the text in 10 sentences that contain the most important information. The text is: \n {}".format(raw_text)}]
        else:
            chat = [{"role": "user", "content": "You will extract the subject-predicate-object triplets from the text and return them in the form [(subject_1; predicate_1; object_1), (subject_2; predicate_2; object_2), ..., (subject_n; predicate_n; object_n)]. We want the subjects and objects in the triplets to be specific, they cannot be pronouns or generic nouns. The first text is: \n {}".format(raw_text)}]

    prompt = ''
    for item in chat:
        if item['role'] == 'user':
            prompt += f'GPT4 Correct User: {item["content"]}<|end_of_turn|>'
        else:
            prompt += f'GPT4 Correct Assistant: {item["content"]}<|end_of_turn|>'

    if output is not None:
        prompt += f'GPT4 Correct Assistant: {output}'
    else:
        prompt += f'GPT4 Correct Assistant:'

    return prompt

def get_prompt_string(raw_text, tokenizer, output=None, few_shot_examples=None, use_fewshot_prompting=True, summarize=False):
    """
    Create a prompt string for the model (not Starling)

    Args:
    raw_text: str: The text to be used in the prompt
    tokenizer: AutoTokenizer: The tokenizer to be used
    output: str: The output to be used in the prompt
    few_shot_examples: list: The few shot examples to be used
    use_fewshot_prompting: bool: Whether to use fewshot prompting
    summarize: bool: Whether to summarize the text

    Returns:
    str: The prompt string
    """

    if use_fewshot_prompting:
        chat = few_shot_examples.copy()
        chat.append({"role": "user", "content": "The next text is: \n {}".format(raw_text)})
    else:
        if summarize:
            chat = [{"role": "user", "content": "Make a summary of the text in 10 sentences that contain the most important information. The text is: \n {}".format(raw_text)}]
        else:
            chat = [{"role": "user", "content": "You will extract the subject-predicate-object triplets from the text and return them in the form [(subject_1; predicate_1; object_1), (subject_2; predicate_2; object_2), ..., (subject_n; predicate_n; object_n)]. We want the subjects and objects in the triplets to be specific, they cannot be pronouns or generic nouns. The first text is: \n {}".format(raw_text)}]
        
    if output is not None:
        chat.append({"role": "assistant", "content": output})
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    else:
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return prompt

def get_prompt(i, df, tokenizer, max_input_length, few_shot_examples, starling=False, use_fewshot_prompting=True):
    """
    Get the prompt string for a given index in the dataframe

    Args:
    i: int: The index of the text in the dataframe
    df: pd.DataFrame: The dataframe containing the texts
    tokenizer: AutoTokenizer: The tokenizer to be used
    max_input_length: int: The maximum input length
    few_shot_examples: list: The few shot examples to be used
    starling: bool: Whether the model is Starling
    use_fewshot_prompting: bool: Whether to use fewshot prompting

    Returns:
    str: The prompt string
    """

    text = df.iloc[i]['input_text']    
    output = df.iloc[i]['labels']
    if starling:
        prompt = get_prompt_string_starling(text, tokenizer, output, use_fewshot_prompting=use_fewshot_prompting, few_shot_examples=few_shot_examples)
    else:
        prompt = get_prompt_string(text, tokenizer, output, few_shot_examples, use_fewshot_prompting)
    return prompt

def load_test_data_csv(path):
    """
    Load the test data from a csv file, this is the case when we use the small sample of 20 test sentences in the Data folder

    Args:
    path: Path: The path to the csv file

    Returns:
    pd.DataFrame: The dataframe containing the test data
    """

    df = pd.read_csv(path, header=None, sep=',', names=['input_text', 'labels'])
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def prepare_train_data(df):
    """
    Prepare the training data by replacing the commas with semicolons within the brackets

    Args:
    df: pd.DataFrame: The dataframe containing the training data

    Returns:
    pd.DataFrame: The modified dataframe
    """

    for idx, row in df.iterrows():
        # Now we want to replace the , with ; only within the brackets
        labels = row['labels']
        # replace the , with ; inside the brackets, not outside
        labels = regex.sub(r'(?<=\([^)]*),(?=[^()]*\))', ';', labels)
        df.at[idx, 'labels'] = labels
    return df

def load_dataframe(path_regular):
    """
    Load the training data from the csv files

    Args:
    path_regular: Path: The path to the regular training data

    Returns:
    pd.DataFrame: The dataframe containing the training data
    """

    df = pd.read_csv(path_regular, header=None, sep=',', names=['input_text', 'labels'])
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = prepare_train_data(df)
    return df

def load_test_data(path):
    """
    Load the test data from a folder, this is the case when we use a folder with txt files (e.g. arxiv papers)

    Args:
    path: Path: The path to the folder containing the test data

    Returns:
    list: The list of test files
    list: The list of names of the test files
    """
    # get all the test files in a list, test files are .txt
    test_files = []
    names = []
    for file in path.rglob('*.txt'):
        test_files.append(file)
        # get the name of the file, in the format name.txt
        names.append(file.name)
    return test_files, names

def prepare_test_data(test_files, size_per_batch, subset=1, logger=None):
    """
    Prepare the test data by splitting it into chunks of size size_per_batch
    
    Args:
    test_files: list: The list of test files
    size_per_batch: int: The size of each batch
    subset: int: The subset of the test files to be used
    logger: Logger: The logger to be used
    
    Returns:
    list: The list of test data
    """

    # split the text into chunks of max_len, split on \n
    test_data = []

    for file in test_files:
        subfiles = []
        with open(file, 'r') as f:
            # read the lines
            lines = f.readlines()


        # split it into chunks of lines size_per_batch
        for i in range(0, len(lines), size_per_batch):
            # make a string with the lines, separated by \n
            text = ''.join(lines[i:i+size_per_batch])
            subfiles.append(text)
            
        test_data.append(subfiles)
    return test_data


def find_max_length(df_train):
    """
    Find the maximum length of the input and output in the training data

    Args:
    df_train: pd.DataFrame: The training data

    Returns:
    int: The maximum length of the input
    int: The maximum length of the output
    """

    max_len_input = 0
    max_len_output = 0
    for i in range(len(df_train)):
        if len(df_train.iloc[i]['input_text']) > max_len_input:
            max_len_input = len(df_train.iloc[i]['input_text'])

        if len(df_train.iloc[i]['labels']) > max_len_output:
            max_len_output = len(df_train.iloc[i]['labels'])
    return max_len_input, max_len_output

def load_model(model_id):
    """
    Load the model

    Args:
    model_id: str: The model id

    Returns:
    AutoModelForCausalLM: The model
    """

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation = 'flash_attention_2')
    return model

def main():
    """
    The main function, run the triplet extraction task using the configured settings
    """

    #################### SETTINGS ####################
    
    MAX_NEW_TOKENS = configs_llm_extraction.MAX_NEW_TOKENS
    DO_SAMPLE = configs_llm_extraction.DO_SAMPLE
    TRAIN_MODEL = configs_llm_extraction.TRAIN_MODEL
    NUM_LINES_PER_BATCH = configs_llm_extraction.NUM_LINES_PER_BATCH
    BATCH_SIZE = configs_llm_extraction.BATCH_SIZE
    QUANTIZE = configs_llm_extraction.QUANTIZE
    USE_FEWSHOT_PROMPTING = configs_llm_extraction.USE_FEWSHOT_PROMPTING
    USE_FINE_TUNE_MODEL = configs_llm_extraction.USE_FINE_TUNE_MODEL
    STARTING_POINT = configs_llm_extraction.STARTING_POINT

    if DO_SAMPLE:
        REPETITION_PENALTY = configs_llm_extraction.REPETITION_PENALTY
        TEMPERATURE = configs_llm_extraction.TEMPERATURE
        TOP_P = configs_llm_extraction.TOP_P

    if TRAIN_MODEL:
        NUM_EPOCHS = configs_llm_extraction.NUM_EPOCHS
        LEARNING_RATE = configs_llm_extraction.LEARNING_RATE

    USE_FEWSHOT_TRAINDATA = configs_llm_extraction.USE_FEWSHOT_TRAINDATA
    PATH_TRAIN_DATASET = configs_llm_extraction.PATH_TRAIN_DATASET

    PATH_FEW_SHOT_EXAMPLES = configs_llm_extraction.PATH_FEW_SHOT_EXAMPLES
    PATH_TEST_DATA = configs_llm_extraction.PATH_TEST_DATA
    PATH_LOG = configs_llm_extraction.PATH_LOGS

    model_id = sys.argv[1]
    if model_id == "mistralai/Mistral-7B-Instruct-v0.2":
        OUTPUT_DIR = configs_llm_extraction.OUTPUT_DIR.joinpath('mistral')
    elif model_id == "CohereForAI/c4ai-command-r-v01":
        OUTPUT_DIR = configs_llm_extraction.OUTPUT_DIR.joinpath('command')
    elif model_id == "Nexusflow/Starling-LM-7B-beta":
        OUTPUT_DIR = configs_llm_extraction.OUTPUT_DIR.joinpath('starling')
    elif model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
        OUTPUT_DIR = configs_llm_extraction.OUTPUT_DIR.joinpath('llama3')
    else:
        OUTPUT_DIR = configs_llm_extraction.OUTPUT_DIR.joinpath(str(model_id))
    
    PATH_SAVE_TEST_DATA = OUTPUT_DIR.joinpath('triplets')

    ##################################################

    # open the fewshot examples
    with open(PATH_FEW_SHOT_EXAMPLES, 'r') as f:
        few_shot_examples = json.load(f)

    if not PATH_LOG.parent.exists():
        PATH_LOG.parent.mkdir(parents=True)

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    
    if not PATH_SAVE_TEST_DATA.exists():
        PATH_SAVE_TEST_DATA.mkdir(parents=True)

    logger = get_logger('general_peft_logger', PATH_LOG)

    #################### LOAD DATA ####################
    logger.info('Loading data...')
    df_train = load_dataframe(PATH_TRAIN_DATASET)
    logger.info('Number of regular training samples: ' + str(len(df_train)))

    max_input_length, max_output_length = find_max_length(df_train)
    logger.info(f'Max input length: {max_input_length}')
    prompt_length = max_input_length - 1

    # check whether path to test data is a csv or a folder
    if PATH_TEST_DATA.suffix == '.csv':
        df_test = load_test_data_csv(PATH_TEST_DATA)
        test_files = [df_test['input_text'].tolist()]
        names = ['test_set_output.txt']
    else:
        test_files, names = load_test_data(PATH_TEST_DATA)
        test_files = prepare_test_data(test_files, NUM_LINES_PER_BATCH, logger=logger)

    # #################### LOAD TOKENIZER AND MODEL ####################
    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=max_input_length)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    #################### MAKE PROMPTS FROM INPUT DATA ####################
    if model_id == "Nexusflow/Starling-LM-7B-beta":
        train_prompts_regular = [get_prompt(i, df_train, tokenizer, max_input_length, few_shot_examples=few_shot_examples, starling=True, use_fewshot_prompting=USE_FEWSHOT_PROMPTING) for i in range(len(df_train))]
        train_prompts_fewshot = [get_prompt_fewshot(i, df_fewshot, tokenizer, max_input_length, few_shot_examples=few_shot_examples, starling=True) for i in range(len(df_fewshot))]
    else:
        train_prompts_regular = [get_prompt(i, df_train, tokenizer, max_input_length, few_shot_examples=few_shot_examples, use_fewshot_prompting=USE_FEWSHOT_PROMPTING) for i in range(len(df_train))]
        train_prompts_fewshot = [get_prompt_fewshot(i, df_fewshot, tokenizer, max_input_length, few_shot_examples=few_shot_examples) for i in range(len(df_fewshot))]

    train_prompts = train_prompts_regular
    train_dataset = Dataset.from_dict({"text": train_prompts})

    if model_id == "Nexusflow/Starling-LM-7B-beta":
        test_prompts = []
        test_texts = []
        for i, file in enumerate(test_files):
            test_prompts_temp = []
            test_texts_temp = []
            for j, text in enumerate(file):
                test_prompts_temp.append(get_prompt_string_starling(text, tokenizer, use_fewshot_prompting=USE_FEWSHOT_PROMPTING, few_shot_examples=few_shot_examples))
                test_texts_temp.append((text, names[i]))
            test_prompts.append(test_prompts_temp)
            test_texts.append(test_texts_temp)
            
    else:
        test_prompts = []
        test_texts = []
        for i, file in enumerate(test_files):
            test_prompts_temp = []
            test_texts_temp = []
            for j, text in enumerate(file):
                test_prompts_temp.append(get_prompt_string(text, tokenizer, use_fewshot_prompting=USE_FEWSHOT_PROMPTING, few_shot_examples=few_shot_examples))
                test_texts_temp.append((text, names[i]))
            test_prompts.append(test_prompts_temp)
            test_texts.append(test_texts_temp)

    logger.info('Number of test files: ' + str(len(test_prompts)))

    lengths = [len(prompt) for sublist in test_prompts for prompt in sublist]
    percentile = np.percentile(lengths, 90)

    device_map = 'auto'

    if TRAIN_MODEL:
        if QUANTIZE:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map, # to use max gpu resources if exist
                quantization_config=bnb_config,
                attn_implementation = 'flash_attention_2',
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,   
                device_map=device_map, # to use max gpu resources if exist
                torch_dtype=torch.bfloat16,
                attn_implementation = 'flash_attention_2',
            )
            
        #Configure the pad token in the model
        model.config.pad_token_id = tokenizer.pad_token_id

        # Define LoRA Config
        lora_config = LoraConfig(   
            lora_alpha=64,
            lora_dropout=0.1,
            r=32,
            bias="none",
            task_type="CAUSAL_LM", 
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )

        run_name = "train mistral-7b"
        training_arguments = TrainingArguments(
            output_dir="./models/"+run_name,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4, # Increase, if still giving OOM error (original 4)
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            save_steps=500,
            logging_steps=200,
            learning_rate=LEARNING_RATE,
            bf16=True, # Enable fp16, bf16 only if your gfx card supports it
            evaluation_strategy="steps", 
            max_grad_norm=0.3,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.001,
            warmup_steps=50, #original 50
            lr_scheduler_type="linear",
            #run_name=run_name,
            report_to='none',
        )

        #run = wandb.init(project="Mixtral-Inst-7b", name= run_name)
        
        trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=max_input_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        )

        # Train the model
        trainer.train()

        # Save the model
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        logger.info('Training completed...')

        # show nvidia-smi
        os.system('nvidia-smi')

        # Clear everything from the GPU
        torch.cuda.empty_cache()
        del model
        del trainer

    if QUANTIZE:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        # Now evaluate - first load the base model again
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device_map, # to use max gpu resources if exist
            quantization_config=bnb_config,
            attn_implementation = 'flash_attention_2',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device_map, # to use max gpu resources if exist
            torch_dtype=torch.bfloat16,
            attn_implementation = 'flash_attention_2',
        )

    if USE_FINE_TUNE_MODEL:
        # Load the fine-tuned model
        model = PeftModel.from_pretrained(model, OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    total_memory = find_total_usable_memory()
    logger.info('Total memory: ' + str(total_memory))

    if model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]

    #initialize a data frame to store the results, if it already exists, load it, else create a new one
    if PATH_SAVE_TEST_DATA.joinpath('triplets.csv').exists():
        results_df = pd.read_csv(PATH_SAVE_TEST_DATA.joinpath('triplets.csv'))
    else:
        results_df = pd.DataFrame(columns=['text', 'triplets', 'paper_id'])

    generated_texts = []
    times = []
    num_files = len(test_prompts)
    #apply chat template
    chat = [{"role": "user", "content": 'Simplify the subjects and objects of the triplets, and split the triplets into multiple ones if needed.'}]
    #tokenize, return a string
    additional_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    #Remove the <begin_of_text> token
    additional_prompt = additional_prompt.replace('<begin_of_text>', '')

    with torch.inference_mode():
        for i, prompts in enumerate(test_prompts):
            if (i + 1) < STARTING_POINT:
                continue
            logger.info(f'Processing file {i+1} of {len(test_prompts)}')
            name_file = names[i]

            with open(PATH_SAVE_TEST_DATA.joinpath(name_file), 'w') as f:
                input_ids = tokenizer(prompts, return_tensors='pt', padding='longest', truncation=True)
                # log shape of input ids
                logger.info('Input ids shape: ' + str(input_ids['input_ids'].shape))
                num_tokens = input_ids['input_ids'].shape[1]
                estimated_memory = estimate_memory(num_tokens, MAX_NEW_TOKENS)
                logger.info('Estimated memory usage: ' + str(estimated_memory))
                batch_size = int(total_memory / estimated_memory)

                attention_mask = input_ids['attention_mask'].cuda()
                input_ids = input_ids['input_ids'].cuda()
                batch_count = 0
                for j in range(0, len(input_ids), batch_size):
                    start_time = time.time()
                    batch = input_ids[j:j+batch_size]
                    mask = attention_mask[j:j+batch_size]
                    if DO_SAMPLE:
                        if model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
                            output = model.generate(input_ids=batch, max_new_tokens=MAX_NEW_TOKENS, attention_mask=mask, do_sample=True, temperature=temp, eos_token_id=terminators, repetition_penalty=REPETITION_PENALTY, top_p = TOP_P)
                        else:
                            output = model.generate(input_ids=batch, max_new_tokens=MAX_NEW_TOKENS, attention_mask=mask, do_sample=True, temperature=temp, repetition_penalty=REPETITION_PENALTY, top_p = TOP_P)

                    else:
                        if model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
                            output = model.generate(input_ids=batch, max_new_tokens=MAX_NEW_TOKENS, attention_mask=mask, eos_token_id=terminators, return_dict_in_generate=True)
                            decoded = tokenizer.batch_decode(output['sequences'], skip_special_tokens=False)
                            # Now we check whether the final token is <|eot_id|> or not, otherwise we need to add it
                            for k, item in enumerate(decoded):
                                if item[-1] != '<|eot_id|>':
                                    decoded[k] += '<|eot_id|>'

                            # Now we want to add the additional prompt
                            for k, item in enumerate(decoded):
                                decoded[k] += additional_prompt
                                
                            output = tokenizer(decoded, return_tensors='pt', padding='longest', truncation=True)
                            output = model.generate(input_ids=output['input_ids'].cuda(), attention_mask=output['attention_mask'].cuda(), max_new_tokens=MAX_NEW_TOKENS)


                        else:
                            output = model.generate(input_ids=batch, max_new_tokens=MAX_NEW_TOKENS, attention_mask=mask)
                    end_time = time.time()
                    times.append(end_time - start_time)

                    result = tokenizer.batch_decode(output, skip_special_tokens=True)
                    # log the length
                    for k, item in enumerate(result):
                        if model_id == "Nexusflow/Starling-LM-7B-beta":
                            extracted_triplets = item.split('GPT4 Correct Assistant:')[-1]
                        elif model_id == "mistralai/Mistral-7B-Instruct-v0.2":
                            extracted_triplets = item.split('[/INST]')[-1]
                        elif model_id == "google/gemma-2b-it":
                            extracted_triplets = item.split('<start_of_turn>model')[-1]
                        elif model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
                            extracted_triplets = item.split('assistant')[-1]
                        elif model_id == "CohereForAI/c4ai-command-r-v01":
                            extracted_triplets = item.split('<|CHATBOT_TOKEN|>')[-1]
                        else:
                            # throw an error, model not supported
                            raise ValueError('Model not supported, please adapt the code such that the generated text is split correctly')

                        # add to the dataframe
                        temp_df = pd.DataFrame({'text': [test_texts[i][k + batch_count * batch_size][0]], 'triplets': [extracted_triplets], 'paper_id': [test_texts[i][k + batch_count * batch_size][1]]})
                        # concatenate the dataframes
                        results_df = pd.concat([results_df, temp_df], ignore_index=True)

                    del batch
                    del mask
                    torch.cuda.empty_cache()
                    batch_count += 1
                    
            if num_files > 200:
                if i % (num_files // 200) == 0:
                    results_df.to_csv(PATH_SAVE_TEST_DATA.joinpath('triplets.csv'), index=False)
                    logger.info('Results saved intermediately at ' + str(PATH_SAVE_TEST_DATA))

    logger.info('Results saved at ' + str(PATH_SAVE_TEST_DATA))
    avg_time = sum(times) / len(times)
    # one batch is BATCH_SIZE number of texts, each text is NUM_LINES_PER_BATCH, calculate the time per line
    time_per_line = avg_time / (BATCH_SIZE * NUM_LINES_PER_BATCH)
    logger.info(f'Average time per batch: {avg_time} seconds')
    logger.info(f'Average time per line: {time_per_line} seconds')

    # save the results dataframe
    results_df.to_csv(PATH_SAVE_TEST_DATA.joinpath('triplets.csv'), index=False)


if __name__ == '__main__':
    main()

    






