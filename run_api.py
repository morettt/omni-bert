import requests

def check_text(text):
    response = requests.post("https://u183410-9a34-bfb19cdf.westb.seetacloud.com:8443/check", params={"text": text})
    return response.json()

print("欢迎使用视觉需求检测工具！")
print("输入 'q' 退出\n")

while True:
    text = input("请输入要检测的文本: ")
    if text.lower() == 'q':
        print("再见！")
        break
        
    try:
        result = check_text(text)
        print(f"结果：{result['需要视觉']}\n")
    except:
        print("检测失败，请确保服务已启动\n")