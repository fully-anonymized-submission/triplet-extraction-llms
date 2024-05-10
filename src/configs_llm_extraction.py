from pathlib import Path
#-----------------------SETTINGS-----------------------

TRAIN_MODEL = False
MAX_NEW_TOKENS = 400
DO_SAMPLE = False
NUM_LINES_PER_BATCH = 15
BATCH_SIZE = 12
USE_FINE_TUNE_MODEL = False
USE_FEWSHOT_PROMPTING = True
QUANTIZE = False
STARTING_POINT = 0


if DO_SAMPLE:
    REPETITION_PENALTY = 1
    TEMPERATURE = 1
    TOP_P = 0.9

if TRAIN_MODEL:
    NUM_EPOCHS = 1.0
    LEARNING_RATE = 3e-4
    
#-----------------------PATHS-----------------------

PATH_TRAIN_DATASET = Path('data/peft/processed_text/train/traindata_peft.csv')
PATH_FEW_SHOT_EXAMPLES = Path('data/peft/processed_text/train/iterative_prompt.json')
PATH_TEST_DATA = Path('data/peft/processed_text/test/test_data.csv')
PATH_LOGS = Path('data/peft/logs/')
OUTPUT_DIR = Path('data/peft/test_set/revised_prompt')