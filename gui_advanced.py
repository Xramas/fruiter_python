import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import json
import threading

class FruitClassifierApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("水果识别器 (Advanced)")
        self.root.geometry("450x500")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = None
        
        # --- UI Elements ---
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Image display label
        self.panel = ttk.Label(main_frame, text="请选择一张图片进行识别", style="TLabel")
        self.panel.pack(pady=10, fill=tk.BOTH, expand=True)

        # Result label
        self.result_label = ttk.Label(main_frame, text="识别结果:", font=("Helvetica", 14), style="TLabel")
        self.result_label.pack(pady=10)
        
        # Button
        self.btn = ttk.Button(main_frame, text="选择图片", command=self.start_prediction_thread)
        self.btn.pack(pady=5)

        # --- Load Resources ---
        try:
            self.load_resources()
        except FileNotFoundError as e:
            messagebox.showerror("错误", f"加载资源失败: {e}\n请确保 'best_model.pt' 和 'classes.json' 文件存在。")
            self.btn.config(state=tk.DISABLED)

    def load_resources(self):
        """Loads the model architecture, trained weights, and class names."""
        print("Loading resources...")
        with open("classes.json", "r", encoding="utf-8") as f:
            self.classes = json.load(f)
        
        num_classes = len(self.classes)
        
        # Load the same architecture as used in training
        self.model = models.efficientnet_v2_s(weights=None) # No pre-trained weights needed here
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Load the trained weights
        self.model.load_state_dict(torch.load("best_model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Image transformation pipeline (must match validation transform)
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Resources loaded successfully.")

    def start_prediction_thread(self):
        """Starts a new thread for prediction to keep the GUI responsive."""
        filepath = filedialog.askopenfilename(
            title="选择一张水果图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not filepath:
            return
            
        self.btn.config(state=tk.DISABLED)
        self.result_label.config(text="正在识别中...")
        
        # Run prediction in a separate thread
        thread = threading.Thread(target=self.predict_image, args=(filepath,))
        thread.start()

    def predict_image(self, filepath):
        """Handles image processing and model prediction."""
        try:
            # Display the selected image
            img_pil = Image.open(filepath).convert("RGB")
            img_tk = ImageTk.PhotoImage(img_pil.resize((350, 350)))
            self.panel.config(image=img_tk)
            self.panel.image = img_tk

            # Pre-process the image for the model
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

            # Perform prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
                predicted_class = self.classes[pred_idx.item()]
                confidence_score = confidence.item()

            result_text = f"识别结果: {predicted_class} (置信度: {confidence_score:.2%})"
            self.result_label.config(text=result_text)

        except Exception as e:
            messagebox.showerror("预测错误", f"无法识别图片: {e}")
            self.result_label.config(text="识别失败")
        finally:
            # Re-enable the button
            self.btn.config(state=tk.NORMAL)

if __name__ == '__main__':
    root = tk.Tk()
    app = FruitClassifierApp(root)
    root.mainloop()