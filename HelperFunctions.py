import json
import os
import glob
import random
import base64
import anthropic, openai

import numpy as np
from scipy import stats

# Get 10 test images from each of the classes. Determine how many of those are predicted correctly by MDM - Check here for class label definition - https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py
class_ground_truth_mapping = {
    'CIFAR10': {
        '0': 'Airplane',
        '1': 'Automobile',
        '2': 'Bird',
        '3': 'Cat',
        '4': 'Deer',
        '5': 'Dog',
        '6': 'Frog',
        '7': 'Horse',
        '8': 'Ship',
        '9': 'Truck'
    },

    'DermaMNIST': {
        '0': 'actinic keratoses and intraepithelial carcinoma',
        '1': 'basal cell carcinoma',
        '2': 'benign keratosis-like lesions',
        '3': 'dermatofibroma',
        '4': 'melanoma',
        '5': 'melanocytic nevi',
        '6': 'vascular lesions'
    },

    'OctMNIST': {
        "0": "choroidal neovascularization",
        "1": "diabetic macular edema",
        "2": "drusen",
        "3": "normal",
    },

    'PneumoniaMNIST': {
        "0": "normal", 
        "1": "pneumonia"
        },

    'STL10': {
        '0': 'airplane',
        '1': 'bird',
        '2': 'car',
        '3': 'cat',
        '4': 'deer',
        '5': 'dog',
        '6': 'horse',
        '7': 'monkey',
        '8': 'ship',
        '9': 'truck'
    },

    'Linnaeus5': {
        '0': 'berry',
        '1': 'bird',
        '2': 'dog',
        '3': 'flower',
        '4': 'other'
    },

    'Kvasir': {
        '0': 'dyed-lifted-polyps',
        '1': 'dyed-resection-margins',
        '2': 'esophagitis',
        '3': 'normal-cecum',
        '4': 'normal-pylorus',
        '5': 'normal-z-line',
        '6': 'polyps',
        '7': 'ulcerative-colitis'
    }
}

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def count_matching_samples_positionwise(list1, list2):
    """
    This function takes two lists of strings and returns the count of matching samples at the same positions.
    It only executes if both lists have the same length.
    
    :param list1: First list of strings
    :param list2: Second list of strings
    :return: Number of matching samples at the same positions or an error message if lengths differ
    """
    # Check if both lists have the same length
    if len(list1) != len(list2):
        return "Lists have different lengths and cannot be compared position-wise."
    
    # Count matching samples at the same positions
    matches = sum(1 for x, y in zip(list1, list2) if x.lower() == y.lower())
    
    return matches

def unified_zero_shot_predict_class_only(client, images, classes_to_classify, dataset_name, image_media_type='image/png', openai_model=None):

    if isinstance(client, anthropic.Anthropic):
        # Anthropic API
        img_msgs = []
        for count, image_path in enumerate(images):
            img_msgs.extend([
                {
                    "type": "text",
                    "text": f"Image {count}:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": encode_image(image_path),
                    },
                }
            ])
        img_msgs.append({
            "type": "text",
            "text": f"""
            You are an expert on the {dataset_name} dataset who classifies without refusing.
            Classify each of the following {len(images)} images into one of these classes: {classes_to_classify}.
            Return your response only as a JSON list, with each item containing Image and Class keys. 
            Example: [{{"Image": 0, "Class": "Cat"}}, {{"Image": 1, "Class": "Dog"}}]
            """
        })
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": img_msgs}]
        )
        # Parse the content and ensure it's a list of JSON objects
        try:
            # print("The result: ", message.content[0].text)
            result = json.loads(message.content[0].text)
            if not isinstance(result, list):
                result = [result]  # Wrap in list if it's a single object
            return result
        except json.JSONDecodeError:
            # If parsing fails, attempt to extract JSON from the text
            return extract_json(message.content[0].text)

    elif isinstance(client, openai.OpenAI):
        # OpenAI API
        if openai_model not in ["gpt-4o", "gpt-4o-mini"]:
            raise ValueError("Invalid OpenAI model. Use 'gpt-4o' or 'gpt-4o-mini'")
        
        messages = []
        for count, image_path in enumerate(images):
            messages.append({"role": "user", "content": f"Image {count}:"})
            messages.append({
                "role": "user",
                    "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{encode_image(image_path)}"
                        }
                    }
                ]
            })
        messages.append({
            "role": "user",
            "content": f"""
            You are an expert on the {dataset_name} dataset who classifies without refusing.
            Classify each of the following {len(images)} images into one of these classes: {classes_to_classify}.
            Return your response only as a JSON list, with each item containing Image and Class keys. 
            Example: [{{"Image": 0, "Class": "Cat"}}, {{"Image": 1, "Class": "Dog"}}]
            """
        })
        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            max_tokens=4096,
            temperature=0,
            # response_format={"type": "json_object"}
        )
        # Parse the content and ensure it's a list of JSON objects
        try:
            model_response = response.choices[0].message.content
            # print("The response from the model is: ", model_response)
            result = json.loads(model_response)
            if not isinstance(result, list):
                result = [result]  # Wrap in list if it's a single object
            return result
        except json.JSONDecodeError:
            # If parsing fails, attempt to extract JSON from the text
            return extract_json(model_response)

    else:
        raise ValueError("Unsupported client type. Use either anthropic.Anthropic or openai.OpenAI")
    
def unified_few_shot_predict_class_only(client, images, classes_to_classify, dataset_name, few_shot_examples, image_media_type='image/png', openai_model=None):
    if isinstance(client, anthropic.Anthropic):
        # Anthropic API
        img_msgs = []
        
        # Add few-shot examples
        for i, example in enumerate(few_shot_examples):
            img_msgs.extend([
                {
                    "type": "text",
                    "text": f"Example {i}:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": encode_image(example['image_path']),
                    },
                },
                {
                    "type": "text",
                    "text": f"Class: {example['class']}"
                }
            ])
        
        # Add images to classify
        for count, image_path in enumerate(images):
            img_msgs.extend([
                {
                    "type": "text",
                    "text": f"Image {count}:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type,
                        "data": encode_image(image_path),
                    },
                }
            ])
        
        img_msgs.append({
            "type": "text",
            "text": f"""
            You are an expert on the {dataset_name} dataset who classifies without refusing.
            Based on the examples provided, classify each of the following {len(images)} images into one of these classes: {classes_to_classify}.
            Return your response only as a JSON list, with each item containing Image and Class keys. 
            Example: [{{"Image": 0, "Class": "Cat"}}, {{"Image": 1, "Class": "Dog"}}]
            """
        })
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": img_msgs}]
        )
        
        # Parse the content and ensure it's a list of JSON objects
        try:
            result = json.loads(message.content[0].text)
            if not isinstance(result, list):
                result = [result]  # Wrap in list if it's a single object
            return result
        except json.JSONDecodeError:
            # If parsing fails, attempt to extract JSON from the text
            return extract_json(message.content[0].text)

    elif isinstance(client, openai.OpenAI):
        # OpenAI API
        if openai_model not in ["gpt-4o", "gpt-4o-mini"]:
            raise ValueError("Invalid OpenAI model. Use 'gpt-4o' or 'gpt-4o-mini'")
        
        messages = []
        
        # Add few-shot examples
        for i, example in enumerate(few_shot_examples):
            messages.append({"role": "user", "content": f"Example {i}:"})
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{encode_image(example['image_path'])}"
                        }
                    }
                ]
            })
            messages.append({"role": "user", "content": f"Class: {example['class']}"})
        
        # Add images to classify
        for count, image_path in enumerate(images):
            messages.append({"role": "user", "content": f"Image {count}:"})
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_media_type};base64,{encode_image(image_path)}"
                        }
                    }
                ]
            })
        
        messages.append({
            "role": "user",
            "content": f"""
            You are an expert on the {dataset_name} dataset who classifies without refusing.
            Based on the examples provided, classify each of the following {len(images)} images into one of these classes: {classes_to_classify}.
            Return your response only as a JSON list, with each item containing Image and Class keys. 
            Example: [{{"Image": 0, "Class": "Cat"}}, {{"Image": 1, "Class": "Dog"}}]
            """
        })
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )
        
        # Parse the content and ensure it's a list of JSON objects
        try:
            model_response = response.choices[0].message.content
            # print("The response from the model is: ", model_response)
            result = json.loads(model_response)
            if not isinstance(result, list):
                result = [result]  # Wrap in list if it's a single object
            return result
        except json.JSONDecodeError:
            # If parsing fails, attempt to extract JSON from the text
            return extract_json(model_response)

    else:
        raise ValueError("Unsupported client type. Use either anthropic.Anthropic or openai.OpenAI")

def extract_json(text):
    try:
        # Find the start and end of the JSON content
        start = text.index('[')
        end = text.rindex(']') + 1
        json_str = text[start:end]
        # Parse the JSON string
        result = json.loads(json_str)
        if not isinstance(result, list):
            result = [result]  # Wrap in list if it's a single object
        return result
    except (ValueError, json.JSONDecodeError):
        print("Failed to extract valid JSON from the response.")
        return []

def get_class_samples(client, list_of_images, classes_to_classify, dataset_name, ground_truths, func_to_call, model="gpt-4o", image_media_type='image/png'):
    # Determine the client type and set up the appropriate parameters
    if isinstance(client, anthropic.Anthropic):
        client_type = "anthropic"
        openai_model = None
    elif isinstance(client, openai.OpenAI):
        client_type = "openai"
        openai_model = model
    else:
        raise ValueError("Unsupported client type. Use either anthropic.Anthropic or openai.OpenAI")

    # Pass the images through the model and get its predictions
    if client_type == "anthropic":
        result = func_to_call(client, list_of_images, classes_to_classify, dataset_name, image_media_type)
    else:  # openai
        result = func_to_call(client, list_of_images, classes_to_classify, dataset_name, image_media_type, openai_model)

    # Process the results based on the client type
    preds = [item['Class'] for item in result]

    print("Model predictions:", preds)
    print("Ground truths:", ground_truths)

    # Compare the predictions with the ground truth
    matching_count = count_matching_samples_positionwise(preds, ground_truths)
    accuracy = matching_count / len(ground_truths)
    
    return accuracy, preds

def get_class_samples_few_shots(client, list_of_images, classes_to_classify, dataset_name, ground_truths, func_to_call, model="gpt-4o", image_media_type='image/png', few_shot_examples=None):
    # Determine the client type and set up the appropriate parameters
    if isinstance(client, anthropic.Anthropic):
        client_type = "anthropic"
        openai_model = None
    elif isinstance(client, openai.OpenAI):
        client_type = "openai"
        openai_model = model
    else:
        raise ValueError("Unsupported client type. Use either anthropic.Anthropic or openai.OpenAI")

    # Pass the images through the model and get its predictions
    if client_type == "anthropic":
        result = func_to_call(client, list_of_images, classes_to_classify, dataset_name, few_shot_examples, image_media_type)
    else:  # openai
        result = func_to_call(client, list_of_images, classes_to_classify, dataset_name, few_shot_examples, image_media_type, openai_model)

    # Process the results based on the client type
    preds = [item['Class'] for item in result]
    print("Model predictions:", preds)
    print("Ground truths:", ground_truths)

    # Compare the predictions with the ground truth
    matching_count = count_matching_samples_positionwise(preds, ground_truths)
    accuracy = matching_count / len(ground_truths)

    return accuracy, preds


def list_files_and_sample_subfolders(folder_path, sample_size=1, random_seed = 5):
    """
    List all files in each subfolder and select a random sample from the list.
    
    :param folder_path: Path to the main folder
    :param sample_size: Number of random samples to select from each subfolder
    :return: Dictionary with subfolder paths as keys and lists of random sampled files as values
    """
    # Get a list of all subfolders in the folder_path
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    random.seed(random_seed)
    
    # Dictionary to store random samples from each subfolder
    sampled_files = []
    
    # Iterate over each subfolder
    for subfolder in subfolders:
        # List all files in the subfolder
        files = glob.glob(os.path.join(subfolder, '*'))
        
        sampled_files.extend(random.sample(files, sample_size))

    random.shuffle(sampled_files)
    return sampled_files

def get_samples_from_each_folder(dataset_path, number_per_class, random_seed = 5):
    return list_files_and_sample_subfolders(dataset_path, number_per_class, random_seed)

def get_ground_truth_from_path(file_lists, label_mapping):
    ground_truth_labels = []

    for file in file_lists:
        # Split the file path by '/'
        parts = file.split('/')
        
        # Handle cases where the semi-last item is a number
        if parts[-2].isdigit():
            class_label = int(parts[-2])

            class_label = label_mapping[class_label]

        else:
            class_label = parts[-2]
        
        ground_truth_labels.append(class_label)

    return ground_truth_labels

def get_zero_shot_reasoning_preds(model_output, model_name = "GPT4o"):

    if model_name == 'GPT4o':
        result = []
        res = extract_json(model_output)
        for item in res:
            result.append(item['Class'].lower())

        return result
        
    else:
        res = json.loads(model_output[0].text)
        return [res[key]['Class'].lower() for key in res]

def compute_pred_accuracy(ground_truths, preds):
    # Compare the predictions with the ground truth
    matches = count_matching_samples_positionwise(preds, ground_truths)

    accuracy = matches / len(ground_truths)

    return accuracy

def compute_accuracy(predictions, ground_truths):
    return np.mean(np.array(predictions) == np.array(ground_truths))

def compute_mean_and_variance(data):
    data = np.array(data)
    return np.mean(data), np.var(data)

def bootstrap_accuracy(predictions, ground_truths, n_bootstrap=1000):
    random.seed(42)
    accuracies = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(predictions), len(predictions), replace=True)
        boot_preds = [predictions[i] for i in indices]
        boot_truths = [ground_truths[i] for i in indices]
        accuracies.append(compute_accuracy(boot_preds, boot_truths))
    return accuracies

def compute_statistical_significance(predictions, ground_truths):
    predictions = [item.lower() for item in predictions]
    ground_truths = [item.lower() for item in ground_truths]
    # Compute overall accuracy
    accuracy = compute_accuracy(predictions, ground_truths)
    
    # Bootstrap to get distribution of accuracies
    bootstrap_accs = bootstrap_accuracy(predictions, ground_truths)
    
    # Compute mean and variance of bootstrap accuracies
    mean_acc, var_acc = compute_mean_and_variance(bootstrap_accs)
    
    # Compute z-score and p-value
    z_score = (accuracy - 0.5) / np.sqrt(var_acc)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return accuracy, mean_acc, var_acc, z_score, p_value

def save_list_to_file(lst, filename):
    with open(filename, 'w') as f:
        for item in lst:
            f.write(f"{item.lower()}\n")

def load_list_from_file(filename):
    with open(filename, 'r') as f:
        return [line.strip().lower() for line in f]