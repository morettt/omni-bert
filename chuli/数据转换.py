import json
import re

def convert_txt_to_json(input_file, output_file):
    """将TXT文件转换为训练和验证所需的JSON格式"""
    train_data = []
    test_data = []
    
    # 读取并处理文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取训练集的问题和标签
    train_pattern = r'训[:：]\s*(.*?)\s*([01])$'
    train_matches = re.finditer(train_pattern, content, re.MULTILINE)
    
    for match in train_matches:
        question = match.group(1).strip()
        label = int(match.group(2))
        train_data.append({"text": question, "label": label})
    
    # 使用正则表达式提取验证集的问题和标签
    test_pattern = r'验[:：]\s*(.*?)\s*([01])$'
    test_matches = re.finditer(test_pattern, content, re.MULTILINE)
    
    for match in test_matches:
        question = match.group(1).strip()
        label = int(match.group(2))
        test_data.append({"text": question, "label": label})
    
    # 创建并保存JSON
    json_data = {
        "train_data": train_data,
        "test_data": test_data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {len(train_data)}条训练数据和{len(test_data)}条验证数据已保存到{output_file}")

if __name__ == "__main__":
    input_file = "/root/OmniBERT/放置数据集.txt"
    output_file = "/root/OmniBERT/data/train.json"
    convert_txt_to_json(input_file, output_file)