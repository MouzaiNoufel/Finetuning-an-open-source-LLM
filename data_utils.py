"""
Dataset utilities for loading and preprocessing instruction-following data.
Supports both built-in demo datasets and custom data formats.
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
import config


def load_alpaca_demo_dataset() -> Dataset:
    """
    Load a small demo instruction dataset for testing.
    This creates a synthetic dataset similar to Stanford Alpaca.
    
    Returns:
        Dataset: Hugging Face Dataset object
    """
    demo_data = [
        {
            "instruction": "Explain the concept of machine learning in simple terms.",
            "response": "Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed. Instead of following rigid instructions, machine learning systems identify patterns in data and use those patterns to make predictions or decisions. For example, a machine learning system can learn to recognize cats in photos by studying thousands of cat images."
        },
        {
            "instruction": "Write a Python function to calculate the factorial of a number.",
            "response": "Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError('Factorial is not defined for negative numbers')\n    if n == 0 or n == 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```\n\nThis function handles edge cases and uses iteration to compute the factorial efficiently."
        },
        {
            "instruction": "What are the health benefits of drinking water?",
            "response": "Drinking adequate water provides numerous health benefits: 1) Regulates body temperature through perspiration, 2) Lubricates joints and cushions organs, 3) Aids digestion and prevents constipation, 4) Flushes out toxins through urination, 5) Maintains healthy skin hydration, 6) Supports cardiovascular health by helping the heart pump blood, 7) Improves cognitive function and energy levels, and 8) Helps maintain proper electrolyte balance. Adults should aim for 8-10 glasses daily."
        },
        {
            "instruction": "Describe the process of photosynthesis.",
            "response": "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs primarily in the chloroplasts of plant cells and involves two main stages: 1) Light-dependent reactions: Chlorophyll absorbs sunlight, splitting water molecules to release oxygen and generate ATP and NADPH. 2) Light-independent reactions (Calvin cycle): ATP and NADPH power the conversion of carbon dioxide into glucose. The overall equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂."
        },
        {
            "instruction": "How do I improve my time management skills?",
            "response": "To improve time management: 1) Prioritize tasks using methods like the Eisenhower Matrix (urgent vs important), 2) Set specific, measurable goals with deadlines, 3) Use time-blocking to schedule focused work periods, 4) Eliminate distractions during work sessions, 5) Break large projects into smaller, manageable tasks, 6) Use tools like calendars and to-do lists, 7) Learn to say no to non-essential commitments, 8) Review and adjust your schedule regularly, and 9) Build in buffer time for unexpected events."
        },
        {
            "instruction": "What is the difference between RAM and ROM?",
            "response": "RAM (Random Access Memory) and ROM (Read-Only Memory) are both computer memory types but serve different purposes: RAM is volatile memory that temporarily stores data and programs currently in use; it's fast, allows read/write operations, and loses all data when power is off. ROM is non-volatile memory that permanently stores essential instructions like firmware; it's slower, typically read-only, and retains data without power. RAM determines multitasking capability and performance, while ROM stores the boot instructions needed to start the computer."
        },
        {
            "instruction": "Explain the significance of the greenhouse effect.",
            "response": "The greenhouse effect is a natural process where certain gases in Earth's atmosphere trap heat from the sun, keeping our planet warm enough to support life. Solar radiation passes through the atmosphere, warms the Earth's surface, and is re-emitted as infrared radiation. Greenhouse gases (CO₂, methane, water vapor) absorb this radiation and re-radiate it back to Earth. While essential for life, human activities have increased greenhouse gas concentrations, intensifying this effect and causing global warming, leading to climate change impacts like rising temperatures, melting ice caps, and extreme weather events."
        },
        {
            "instruction": "Write a haiku about autumn.",
            "response": "Crimson leaves falling\nCool breeze whispers through bare trees\nNature's gentle rest"
        },
        {
            "instruction": "What are the main components of a computer?",
            "response": "The main components of a computer are: 1) CPU (Central Processing Unit): the 'brain' that executes instructions, 2) RAM (Random Access Memory): temporary storage for active programs, 3) Storage (HDD/SSD): permanent data storage, 4) Motherboard: connects all components, 5) Power Supply Unit: provides electricity, 6) GPU (Graphics Processing Unit): handles visual processing, 7) Input devices: keyboard, mouse for user interaction, 8) Output devices: monitor, speakers for displaying results, and 9) Cooling system: prevents overheating."
        },
        {
            "instruction": "How does blockchain technology work?",
            "response": "Blockchain is a distributed ledger technology that records transactions across multiple computers. Here's how it works: 1) Transactions are grouped into blocks, 2) Each block contains a cryptographic hash of the previous block, creating a chain, 3) Network nodes validate new blocks through consensus mechanisms (like Proof of Work), 4) Once validated, blocks are added to the chain and distributed across all nodes, 5) The decentralized nature makes it extremely difficult to alter past records, as changes would need to be made across the majority of the network simultaneously. This creates a transparent, secure, and tamper-resistant record system."
        },
        {
            "instruction": "What is the difference between artificial intelligence and machine learning?",
            "response": "Artificial Intelligence (AI) is the broader concept of machines performing tasks that typically require human intelligence, including reasoning, problem-solving, understanding language, and perception. Machine Learning (ML) is a subset of AI that focuses specifically on systems that learn and improve from experience without being explicitly programmed. Think of it this way: AI is the goal (creating intelligent machines), while ML is one method to achieve that goal (teaching machines to learn from data). Other AI approaches include rule-based systems, expert systems, and symbolic reasoning."
        },
        {
            "instruction": "Summarize the plot of Romeo and Juliet.",
            "response": "Romeo and Juliet is Shakespeare's tragic tale of two young lovers from feuding families in Verona, Italy. Romeo Montague and Juliet Capulet meet at a ball and fall instantly in love, marrying secretly with Friar Lawrence's help. When Romeo kills Juliet's cousin Tybalt in a duel and is banished, the couple faces separation. To avoid marrying another man, Juliet takes a potion to feign death, planning to reunite with Romeo. However, Romeo doesn't receive the message explaining the plan, believes Juliet is truly dead, and poisons himself beside her. Juliet awakens, finds Romeo dead, and kills herself. Their deaths finally reconcile the feuding families."
        },
        {
            "instruction": "How do vaccines work?",
            "response": "Vaccines work by training the immune system to recognize and fight specific pathogens without causing the actual disease. Here's the process: 1) A vaccine introduces a weakened, inactive, or partial form of a pathogen (or instructions to make it, as in mRNA vaccines), 2) The immune system detects this foreign substance and produces antibodies and memory cells, 3) These memory cells 'remember' the pathogen, 4) If the real pathogen enters the body later, the immune system quickly recognizes it and mounts a rapid, strong response, 5) This prevents or reduces the severity of infection. Vaccines provide immunity without the risks of the actual disease."
        },
        {
            "instruction": "What is the Pythagorean theorem?",
            "response": "The Pythagorean theorem is a fundamental principle in geometry that describes the relationship between the sides of a right triangle. It states: a² + b² = c², where 'a' and 'b' are the lengths of the two shorter sides (legs) of the right triangle, and 'c' is the length of the longest side (hypotenuse) opposite the right angle. For example, if one leg is 3 units and another is 4 units, the hypotenuse would be 5 units (3² + 4² = 9 + 16 = 25 = 5²). This theorem is attributed to the ancient Greek mathematician Pythagoras and has countless applications in mathematics, physics, engineering, and navigation."
        },
        {
            "instruction": "Explain the water cycle.",
            "response": "The water cycle, or hydrologic cycle, is the continuous movement of water on, above, and below Earth's surface. The main stages are: 1) Evaporation: Heat from the sun causes water from oceans, lakes, and rivers to transform into water vapor, 2) Transpiration: Plants release water vapor through their leaves, 3) Condensation: Water vapor rises, cools, and forms clouds as tiny droplets, 4) Precipitation: When droplets combine and become heavy, they fall as rain, snow, sleet, or hail, 5) Collection: Water accumulates in bodies of water and underground aquifers, 6) Runoff: Water flows over land back to water bodies. This cycle is essential for distributing fresh water across the planet and sustaining all life forms."
        },
        {
            "instruction": "What are the benefits of regular exercise?",
            "response": "Regular exercise provides extensive physical and mental health benefits: Physical benefits include: 1) Strengthens cardiovascular system and reduces heart disease risk, 2) Helps maintain healthy weight and boosts metabolism, 3) Builds muscle strength and bone density, 4) Improves flexibility and balance, 5) Enhances immune function. Mental benefits include: 1) Reduces stress, anxiety, and depression through endorphin release, 2) Improves sleep quality, 3) Enhances cognitive function and memory, 4) Boosts self-esteem and confidence, 5) Increases energy levels. The WHO recommends at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity weekly for adults."
        },
        {
            "instruction": "What is the theory of relativity?",
            "response": "Einstein's theory of relativity consists of two parts: Special Relativity (1905) states that the laws of physics are the same for all non-accelerating observers, and the speed of light is constant regardless of the observer's motion. This leads to time dilation and length contraction at high speeds. General Relativity (1915) extends this to include gravity, describing it not as a force but as a curvature of spacetime caused by mass and energy. Massive objects like planets bend spacetime, and other objects follow curved paths around them, which we perceive as gravitational attraction. This theory revolutionized our understanding of space, time, and gravity, and has been confirmed by numerous experiments."
        },
        {
            "instruction": "How do solar panels generate electricity?",
            "response": "Solar panels generate electricity through the photovoltaic effect: 1) Solar panels contain photovoltaic cells made of semiconductor materials (typically silicon), 2) When sunlight hits the cells, photons transfer energy to electrons in the semiconductor, 3) This energy frees electrons from their atoms, creating electron-hole pairs, 4) The cell's internal electric field directs freed electrons to flow in one direction, creating direct current (DC) electricity, 5) An inverter converts DC to alternating current (AC) for household use, 6) Excess electricity can be stored in batteries or fed back to the power grid. The efficiency depends on factors like panel quality, sunlight intensity, angle, and temperature."
        },
        {
            "instruction": "What is the difference between weather and climate?",
            "response": "Weather and climate are related but distinct concepts: Weather refers to short-term atmospheric conditions in a specific place at a specific time, including temperature, humidity, precipitation, wind, and cloud cover. It changes daily or hourly and can be quite variable. Climate refers to long-term patterns of weather in a region over extended periods (typically 30+ years), representing the average conditions and expected variations. For example, 'it's raining today' describes weather, while 'this region has wet winters and dry summers' describes climate. Think of it this way: climate is what you expect, weather is what you get."
        },
        {
            "instruction": "Explain the concept of compound interest.",
            "response": "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Unlike simple interest (calculated only on principal), compound interest grows exponentially. The formula is: A = P(1 + r/n)^(nt), where A is the final amount, P is principal, r is annual interest rate, n is compounding frequency per year, and t is time in years. For example, $1,000 at 5% annual interest compounded yearly becomes $1,050 after year 1, then $1,102.50 after year 2 (5% of $1,050), and $1,157.63 after year 3. The effect becomes more powerful over time, which Einstein allegedly called 'the eighth wonder of the world.'"
        }
    ]
    
    return Dataset.from_list(demo_data)


def load_custom_dataset(file_path: Path) -> Dataset:
    """
    Load a custom dataset from JSON or CSV file.
    Expected format: Each entry should have 'instruction' and 'response' fields.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Dataset: Hugging Face Dataset object
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .json or .csv")
    
    # Validate required fields
    required_fields = {'instruction', 'response'}
    if data and not required_fields.issubset(set(data[0].keys())):
        raise ValueError(f"Dataset must contain fields: {required_fields}")
    
    return Dataset.from_list(data)


def prepare_dataset() -> DatasetDict:
    """
    Load and prepare the dataset for training.
    Automatically selects demo dataset or custom dataset based on config.
    
    Returns:
        DatasetDict: Dictionary with 'train' and 'test' splits
    """
    print("Loading dataset...")
    
    # Load dataset
    if config.CUSTOM_DATASET_PATH and Path(config.CUSTOM_DATASET_PATH).exists():
        print(f"Loading custom dataset from: {config.CUSTOM_DATASET_PATH}")
        dataset = load_custom_dataset(Path(config.CUSTOM_DATASET_PATH))
    else:
        print("Loading demo Alpaca-style dataset...")
        dataset = load_alpaca_demo_dataset()
    
    print(f"Loaded {len(dataset)} examples")
    
    # Split into train and validation
    split_dataset = dataset.train_test_split(
        test_size=config.TRAIN_TEST_SPLIT,
        seed=config.RANDOM_SEED,
        shuffle=True
    )
    
    print(f"Train examples: {len(split_dataset['train'])}")
    print(f"Validation examples: {len(split_dataset['test'])}")
    
    return split_dataset


def format_instruction(example: Dict[str, str]) -> str:
    """
    Format a single example using the instruction template.
    
    Args:
        example: Dictionary with 'instruction' and 'response' keys
        
    Returns:
        str: Formatted instruction text
    """
    return config.INSTRUCTION_TEMPLATE.format(
        instruction=example['instruction'],
        response=example['response']
    )


def tokenize_function(examples: Dict[str, List], tokenizer: PreTrainedTokenizer) -> Dict:
    """
    Tokenize examples for training.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Dict: Tokenized inputs with input_ids, attention_mask, and labels
    """
    # Format each example
    formatted_texts = [
        config.INSTRUCTION_TEMPLATE.format(
            instruction=instruction,
            response=response
        )
        for instruction, response in zip(examples['instruction'], examples['response'])
    ]
    
    # Tokenize
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        padding='max_length',
        return_tensors=None,
    )
    
    # For causal language modeling, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def prepare_tokenized_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer
) -> DatasetDict:
    """
    Tokenize the entire dataset.
    
    Args:
        dataset: Dataset dictionary with train/test splits
        tokenizer: Hugging Face tokenizer
        
    Returns:
        DatasetDict: Tokenized dataset ready for training
    """
    print("Tokenizing dataset...")
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize datasets
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing dataset",
    )
    
    print("Tokenization complete")
    return tokenized_dataset


def print_dataset_sample(dataset: Dataset, num_samples: int = 2) -> None:
    """
    Print sample examples from the dataset for inspection.
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to print
    """
    print("\n" + "="*80)
    print("DATASET SAMPLES")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        print(f"\nExample {i+1}:")
        print("-" * 80)
        print(f"Instruction: {example['instruction']}")
        print(f"\nResponse: {example['response']}")
        print("-" * 80)
    
    print("\n")
