import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_dataset(json_path):
    """
    从JSON文件加载数据集
    
    Args:
        json_path (str): JSON文件路径
        
    Returns:
        tuple: (train_texts, train_labels, test_texts, test_labels, has_test_data)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_texts = [item['text'] for item in data['train_data']]
    train_labels = [item['label'] for item in data['train_data']]
    
    # 检查是否有测试数据
    has_test_data = False
    test_texts = []
    test_labels = []
    
    if 'test_data' in data and data['test_data']:
        test_texts = [item['text'] for item in data['test_data']]
        test_labels = [item['label'] for item in data['test_data']]
        has_test_data = True
    
    return train_texts, train_labels, test_texts, test_labels, has_test_data

# 创建数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 计算评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_vision_classifier(json_path, model_path="/root/OmniBERT/ernie-3.0-base-zh"):
    """
    训练视觉分类模型
    
    Args:
        json_path (str): 数据集JSON文件路径
        model_path (str): 预训练模型路径
        
    Returns:
        tuple: (model, tokenizer) 训练好的模型和tokenizer
    """
    # 加载数据
    train_texts, train_labels, test_texts, test_labels, has_test_data = load_dataset(json_path)
    print(f"加载了 {len(train_texts)} 条训练数据")
    
    if has_test_data:
        print(f"找到 {len(test_texts)} 条测试数据，将进行评估")
    else:
        print("未找到测试数据，将跳过评估")
    
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        problem_type="single_label_classification"
    )

    # 创建训练数据集
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    
    # 训练参数基础配置
    training_args_dict = {
        "output_dir": "./results",
        "num_train_epochs": 9,
        "per_device_train_batch_size": 8,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_dir": './logs',
        "logging_steps": 10,
        "learning_rate": 2e-5,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 1,
    }
    
    # 如果有测试数据，添加评估相关参数
    if has_test_data:
        eval_dataset = TextDataset(test_texts, test_labels, tokenizer)
        training_args_dict.update({
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "per_device_eval_batch_size": 8,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
        })
    
    # 创建训练参数
    training_args = TrainingArguments(**training_args_dict)

    # 初始化trainer
    trainer_args = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    
    # 如果有测试数据，添加评估数据集和评估函数
    if has_test_data:
        trainer_args.update({
            "eval_dataset": eval_dataset,
            "compute_metrics": compute_metrics
        })
    
    trainer = Trainer(**trainer_args)

    # 训练模型
    print("开始训练模型...")
    trainer.train()
    
    # 如果有测试数据，单独进行一次评估并打印结果
    if has_test_data:
        print("\n进行最终评估...")
        eval_results = trainer.evaluate()
        print(f"评估结果: {eval_results}")

    return model, tokenizer

def predict_vision_required(model, tokenizer, texts):
    """
    预测文本是否需要视觉能力
    
    Args:
        model: 训练好的模型
        tokenizer: 对应的tokenizer
        texts (list): 待预测的文本列表
        
    Returns:
        tuple: (predicted_labels, probabilities) 预测的标签和概率
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = predictions.cpu().numpy()
        predicted_labels = predictions.argmax(dim=-1).cpu().tolist()
        return predicted_labels, probs

if __name__ == "__main__":
    # 数据集
    json_path = "/root/OmniBERT/data/train.json"
    
    # 训练
    model, tokenizer = train_vision_classifier(json_path)
    
    # 检查是否有测试数据进行预测展示
    _, _, test_texts, _, has_test_data = load_dataset(json_path)
    
    if has_test_data and test_texts:
        # 使用测试数据进行预测并显示结果
        print("\n对测试数据进行预测...")
        results, probabilities = predict_vision_required(model, tokenizer, test_texts)
        
        print("\n预测结果示例:")
        max_samples = min(5, len(test_texts))  # 最多显示5个样本
        for i in range(max_samples):
            text = test_texts[i]
            result = results[i]
            prob = probabilities[i]
            print(f"文本: {text}")
            print(f"需要视觉能力: {'是' if result == 1 else '否'}")
            print(f"预测概率: 不需要视觉={prob[0]:.4f}, 需要视觉={prob[1]:.4f}\n")
    
    # 保存
    model_save_path = "/root/OmniBERT/output_models"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"模型已保存到: {model_save_path}")