# Wordpieces | Tokens limits
MAX_TOKENS = 128
MAX_WORDPIECES = 128
MAX_BATCH = 500

# WinoMT data
profession_splits = {"train": {"librarian", "driver", "sheriff", "carpenter", "developer", "assistant", "teacher",
                               "janitor", "cook", "editor", "mechanic", "construction worker", "clerk", "counselor",
                               "auditor", "farmer", "housekeeper", "attendant", "accountant", "laborer", "supervisor",
                               "receptionist", "analyst", "secretary",
                               'protester', 'onlooker', 'client', 'patient', 'undergraduate', 'homeowner', 'practitioner',
                               'someone', 'victim', 'child', 'broker', 'administrator', 'dispatcher', 'educator', 'chemist',
                               'instructor', 'guest', 'technician', 'specialist', 'pathologist', 'passenger',
                               'surgeon', 'programmer', 'witness', 'student', 'doctor', 'electrician', 'advisee',
                               'employee', 'resident', 'paralegal', 'examiner', 'engineer', 'pedestrian', 'psychologist',
                               'teenager', 'hygienist', 'visitor'},
                               
                     "dev": {"physician", "CEO", "baker", "tailor", "cashier", "lawyer", "hairdresser", "chief",
                             'customer', 'investigator', 'appraiser', 'owner', 'pharmacist', 'bartender', 'nutritionist',
                             'chef', 'plumber', 'buyer', 'firefighter', 'dietitian', 'officer'},
                     "test": {"designer", "manager", "guard", "cleaner", "mover", "nurse", "writer", "salesperson",
                              'planner', 'machinist', 'advisor', 'painter', 'taxpayer', 'bystander', 'paramedic',
                              'inspector', 'veterinarian', 'scientist', 'therapist', 'architect', 'worker'}}

male_biased = {"driver", "supervisor", "janitor", "cook", "mover", "laborer", "construction worker", "chief",
               "developer", "carpenter", "manager", "lawyer", "farmer", "salesperson", "physician", "guard",
               "analyst", "mechanic", "sheriff", "CEO",
               "doctor", "programmer", "surgeon",' engineer', 'scientist', 'electrician', 'plumber', 'firefighter',
               'machinist', 'technician', 'officer'}

female_biased = {"attendant", "cashier", "teacher", "nurse", "assistant", "secretary", "auditor", "cleaner",
                 "receptionist", "clerk", "counselor", "designer", "hairdresser", "writer", "housekeeper",
                 "baker", "accountant", "editor", "librarian", "tailor",
                 'hygienist'}

non_biased = {'protester', 'onlooker', 'client', 'patient', 'undergraduate', 'homeowner', 'practitioner', 'someone',
               'victim', 'child', 'broker', 'administrator', 'dispatcher', 'educator', 'chemist', 'instructor', 'guest',
               'specialist', 'pathologist', 'passenger', 'witness', 'student',
                'advisee', 'employee', 'resident', 'paralegal', 'examiner',
               'pedestrian', 'psychologist', 'teenager', 'visitor', 'customer', 'investigator', 'appraiser',
               'owner', 'pharmacist', 'bartender', 'nutritionist', 'chef', 'buyer', 'dietitian',
               'planner', 'advisor', 'painter', 'taxpayer', 'bystander', 'paramedic', 'inspector',
               'veterinarian', 'therapist', 'architect', 'worker' }
                #"doctor", "programmer", "surgeon",' engineer', 'scientist', 'electrician', 'plumber', 'firefighter',
                # 'machinist', 'technician', 'officer', 'hygenist' }

problematic_list = {} #{'doctor', 'programmer', ' engineer', 'scientist', 'electrician', 'plumber', 'firefighter',
                    #'machinist', 'technician', 'specialist', 'surgeon', 'hygienist', 'officer'}

# BERT model parameters
LANGUAGE_CHINESE = "chinese"
LANGUAGE_MULTILINGUAL = "multilingual"

SIZE_BASE = "base"
SIZE_LARGE = "large"

CASING_CASED = "cased"
CASING_UNCASED = "uncased"

SUPPORTED_MODELS = {f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}",
                    f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}",
                    f"bert-{SIZE_BASE}-{CASING_CASED}",
                    f"bert-{SIZE_BASE}-{CASING_UNCASED}",
                    f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}",
                    f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}",
                    f"bert-{SIZE_LARGE}-{CASING_CASED}",
                    f"bert-{SIZE_LARGE}-{CASING_UNCASED}",
                    f"roberta-{SIZE_BASE}",
                    f"roberta-{SIZE_LARGE}",
                    f"xlm-roberta-{SIZE_BASE}",
                    f"xlm-roberta-{SIZE_LARGE}",
                    f"random-bert"
                    }

MODEL_DIMS = {f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}": 768,
              f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}": 768,
              f"bert-{SIZE_BASE}-{CASING_CASED}": 768,
              f"bert-{SIZE_BASE}-{CASING_UNCASED}": 768,
              f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}": 1024,
              f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}": 1024,
              f"bert-{SIZE_LARGE}-{CASING_CASED}": 1024,
              f"bert-{SIZE_LARGE}-{CASING_UNCASED}": 1024,
              f"roberta-{SIZE_BASE}": 768,
              f"roberta-{SIZE_LARGE}": 1024,
              f"xlm-roberta-{SIZE_BASE}": 768,
              f"xlm-roberta-{SIZE_LARGE}": 1024,
              f"random-bert": 768
              }


MODEL_LAYERS = {f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}": 12,
                f"bert-{SIZE_BASE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}": 12,
                f"bert-{SIZE_BASE}-{CASING_CASED}": 12,
                f"bert-{SIZE_BASE}-{CASING_UNCASED}": 12,
                f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_CASED}": 24,
                f"bert-{SIZE_LARGE}-{LANGUAGE_MULTILINGUAL}-{CASING_UNCASED}": 24,
                f"bert-{SIZE_LARGE}-{CASING_CASED}": 24,
                f"bert-{SIZE_LARGE}-{CASING_UNCASED}": 24,
                f"roberta-{SIZE_BASE}": 12,
                f"roberta-{SIZE_LARGE}": 24,
                f"xlm-roberta-{SIZE_BASE}": 12,
                f"xlm-roberta-{SIZE_LARGE}": 24,
                f"random-bert": 12
                }

BERT_MODEL_DIR = "/net/projects/bert/models/"

#data pipeline options
BUFFER_SIZE = 50 * 1000 * 1000
SHUFFLE_SIZE = 512
