"""
Example script showing how to create a custom dataset for training.
This demonstrates the expected format and best practices.
"""

import json
from pathlib import Path


def create_customer_service_dataset():
    """
    Example: Customer service chatbot dataset.
    Replace with your domain-specific data.
    """
    dataset = [
        {
            "instruction": "How do I reset my password?",
            "response": "To reset your password, follow these steps: 1) Click 'Forgot Password' on the login page, 2) Enter your email address, 3) Check your inbox for a reset link, 4) Click the link and enter your new password, 5) Confirm the password and save. If you don't receive the email within 5 minutes, check your spam folder or contact support."
        },
        {
            "instruction": "What is your return policy?",
            "response": "Our return policy allows returns within 30 days of purchase for most items. Items must be unused, in original packaging, and with receipt. To initiate a return: 1) Log into your account, 2) Go to 'Order History', 3) Select the item to return, 4) Choose a reason and submit. You'll receive a prepaid shipping label via email. Refunds are processed within 5-7 business days after we receive the item."
        },
        {
            "instruction": "How can I track my order?",
            "response": "You can track your order in several ways: 1) Log into your account and visit 'Order History', 2) Click on the order number to see detailed tracking information, 3) Use the tracking number sent to your email with the shipping carrier's website, 4) Enable SMS notifications for real-time updates. If tracking information isn't available within 24 hours of shipment, please contact our support team."
        },
    ]
    
    return dataset


def create_code_assistant_dataset():
    """
    Example: Programming assistant dataset.
    """
    dataset = [
        {
            "instruction": "How do I read a CSV file in Python?",
            "response": "You can read a CSV file in Python using the pandas library:\n\n```python\nimport pandas as pd\n\n# Read CSV file\ndf = pd.read_csv('file.csv')\n\n# Display first few rows\nprint(df.head())\n```\n\nAlternatively, use Python's built-in csv module:\n\n```python\nimport csv\n\nwith open('file.csv', 'r') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)\n```"
        },
        {
            "instruction": "Explain list comprehension in Python.",
            "response": "List comprehension is a concise way to create lists in Python. Instead of using loops, you can create lists in a single line:\n\nBasic syntax: [expression for item in iterable]\n\nExample:\n```python\n# Traditional way\nsquares = []\nfor i in range(10):\n    squares.append(i**2)\n\n# List comprehension\nsquares = [i**2 for i in range(10)]\n```\n\nWith condition:\n```python\n# Even numbers only\nevens = [i for i in range(20) if i % 2 == 0]\n```\n\nList comprehensions are faster and more readable for simple transformations."
        },
    ]
    
    return dataset


def create_medical_qa_dataset():
    """
    Example: Medical Q&A dataset.
    Note: For production medical applications, use verified medical sources.
    """
    dataset = [
        {
            "instruction": "What are the common symptoms of the flu?",
            "response": "Common flu symptoms include: 1) Fever or chills (usually high, 100-104°F), 2) Cough (typically dry), 3) Sore throat, 4) Runny or stuffy nose, 5) Muscle or body aches, 6) Headaches, 7) Fatigue and weakness, 8) Some people may experience vomiting and diarrhea (more common in children). Symptoms typically appear suddenly, 1-4 days after exposure. If you experience difficulty breathing, chest pain, or persistent fever above 103°F, seek immediate medical attention."
        },
        {
            "instruction": "How much sleep do adults need?",
            "response": "According to the National Sleep Foundation, most adults (ages 18-64) need 7-9 hours of sleep per night for optimal health. Older adults (65+) may need slightly less, around 7-8 hours. Individual needs vary based on factors like activity level, health conditions, and genetics. Signs you're getting enough sleep include: waking refreshed, maintaining energy throughout the day, and not relying on caffeine. Chronic sleep deprivation can lead to health issues including weakened immunity, weight gain, and increased disease risk."
        },
    ]
    
    return dataset


def save_dataset(dataset, filename):
    """
    Save dataset to JSON file.
    
    Args:
        dataset: List of instruction-response dictionaries
        filename: Output filename
    """
    output_path = Path("data") / filename
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to: {output_path}")
    print(f"Number of examples: {len(dataset)}")


def validate_dataset(dataset):
    """
    Validate dataset format.
    
    Args:
        dataset: List of dictionaries to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(dataset, list):
        print("Error: Dataset must be a list")
        return False
    
    if len(dataset) == 0:
        print("Error: Dataset is empty")
        return False
    
    for idx, example in enumerate(dataset):
        if not isinstance(example, dict):
            print(f"Error at index {idx}: Example must be a dictionary")
            return False
        
        if 'instruction' not in example:
            print(f"Error at index {idx}: Missing 'instruction' field")
            return False
        
        if 'response' not in example:
            print(f"Error at index {idx}: Missing 'response' field")
            return False
        
        if not isinstance(example['instruction'], str):
            print(f"Error at index {idx}: 'instruction' must be a string")
            return False
        
        if not isinstance(example['response'], str):
            print(f"Error at index {idx}: 'response' must be a string")
            return False
        
        if len(example['instruction'].strip()) == 0:
            print(f"Error at index {idx}: 'instruction' is empty")
            return False
        
        if len(example['response'].strip()) == 0:
            print(f"Error at index {idx}: 'response' is empty")
            return False
    
    print(f"✓ Dataset validation passed ({len(dataset)} examples)")
    return True


def main():
    """Create example datasets for different domains."""
    
    print("Creating example datasets...")
    print("="*80)
    
    # Customer service dataset
    print("\n1. Customer Service Dataset")
    cs_dataset = create_customer_service_dataset()
    if validate_dataset(cs_dataset):
        save_dataset(cs_dataset, "customer_service.json")
    
    # Code assistant dataset
    print("\n2. Code Assistant Dataset")
    code_dataset = create_code_assistant_dataset()
    if validate_dataset(code_dataset):
        save_dataset(code_dataset, "code_assistant.json")
    
    # Medical Q&A dataset
    print("\n3. Medical Q&A Dataset")
    medical_dataset = create_medical_qa_dataset()
    if validate_dataset(medical_dataset):
        save_dataset(medical_dataset, "medical_qa.json")
    
    print("\n" + "="*80)
    print("Example datasets created successfully!")
    print("\nTo use a dataset for training:")
    print("1. Edit config.py")
    print("2. Set: CUSTOM_DATASET_PATH = 'data/customer_service.json'")
    print("3. Run: python train.py")
    print("="*80)


if __name__ == "__main__":
    main()
