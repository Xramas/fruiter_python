import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import Net
import json
import threading

class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("水果识别器")
        self.root.geometry("400x450")

        # 加载模型和类别
        self.load_resources()

        # 设置界面
        self.create_widgets()

    def load_resources(self):
        with open("classes.json", "r", encoding="utf-8") as f:
            self.classes = json.load(f)
        
        self.model = Net(num_classes=len(self.classes))
        # 加载训练好的最佳模型
        self.model.load_state_dict(torch.load("best_model.pt")) 
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 图像显示区
        self.panel = ttk.Label(main_frame, text="请选择一张图片", anchor="center")
        self.panel.pack(pady=10, fill=tk.BOTH, expand=True)

        # 结果显示区
        self.result_label = ttk.Label(main_frame, text="识别结果：", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        # 按钮区
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        self.btn = ttk.Button(btn_frame, text="选择图片", command=self.start_prediction_thread)
        self.btn.pack()

    def start_prediction_thread(self):
        # 防止用户在预测期间重复点击
        self.btn.config(state=tk.DISABLED)
        # 启动一个新线程进行预测
        threading.Thread(target=self.predict_image).start()

    def predict_image(self):
        path = filedialog.askopenfilename()
        if not path:
            self.btn.config(state=tk.NORMAL)
            return

        try:
            # 更新图片显示
            img = Image.open(path).convert("RGB")
            tk_img = ImageTk.PhotoImage(img.resize((300, 300)))
            self.panel.config(image=tk_img)
            self.panel.image = tk_img
            self.result_label.config(text="正在识别中...")

            # 预处理和预测
            input_tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
                pred_prob = torch.softmax(output, dim=1)
                pred_score, pred_idx = torch.max(pred_prob, 1)
                
                result_text = f"识别结果：{self.classes[pred_idx.item()]} (置信度: {pred_score.item():.2%})"
                self.result_label.config(text=result_text)

        except Exception as e:
            self.result_label.config(text=f"错误: {e}")
        finally:
            # 预测结束后恢复按钮
            self.btn.config(state=tk.NORMAL)


if __name__ == '__main__':
    root = tk.Tk()
    app = FruitClassifierApp(root)
    root.mainloop()