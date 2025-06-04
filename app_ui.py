import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
from io import StringIO
import sys

# Import the functions from your main script
# Assuming index.py is in the same directory or in Python's path
try:
    from index import perform_training_and_evaluation, perform_prediction_with_saved_models, generate_analysis_figures
except ImportError:
    messagebox.showerror("Import Error", "Could not import functions from index.py. Make sure it's in the same directory.")
    exit()

# Modern color palette
DARK_BG = "#181c2f"
SIDEBAR_GRADIENT_TOP = "#232946"
SIDEBAR_GRADIENT_BOTTOM = "#1a1d2e"
ACCENT_GRADIENT = "#4a90e2"
ACCENT_GRADIENT2 = "#7f53ac"
BUTTON_GRADIENT = "#4a90e2"
BUTTON_GRADIENT2 = "#7f53ac"
GLOW_COLOR = "#f7c873"
SUCCESS_COLOR = "#2ecc71"
WARNING_COLOR = "#e74c3c"
TEXT_COLOR = "#eaeaea"
HEADER_COLOR = "#f7c873"

class ResultsRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.buffer.write(text)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

    def __del__(self):
        sys.stdout = self.old_stdout

class VoiceGenderApp:
    def __init__(self, master):
        self.master = master
        master.title("Voice Gender Recognition")
        master.geometry("1400x900")
        master.configure(bg=DARK_BG)

        # Sidebar with gradient and shadow
        self.sidebar = tk.Canvas(master, width=600, highlightthickness=0, bg=SIDEBAR_GRADIENT_TOP)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        self.sidebar.create_rectangle(0, 0, 600, 900, fill=SIDEBAR_GRADIENT_TOP, outline="")
        for i in range(0, 900, 2):
            color = f"#{int(35 + (26-35)*i/900):02x}{int(41 + (29-41)*i/900):02x}{int(70 + (46-70)*i/900):02x}"
            self.sidebar.create_rectangle(0, i, 600, i+2, fill=color, outline="")
        self.sidebar_frame = tk.Frame(self.sidebar, bg=SIDEBAR_GRADIENT_TOP)
        self.sidebar.create_window((0, 0), window=self.sidebar_frame, anchor="nw", width=600, height=900)

        # User info at the top with shadow/glow
        self.header_info = tk.Frame(self.sidebar_frame, bg=SIDEBAR_GRADIENT_TOP)
        self.header_info.pack(fill=tk.X, pady=(18, 0))
        self.header_name = tk.Label(
            self.header_info,
            text="Mohammad Tosifi",
            font=("Segoe UI", 15, "bold"),
            bg=SIDEBAR_GRADIENT_TOP,
            fg=HEADER_COLOR,
            highlightthickness=0
        )
        self.header_name.pack(anchor=tk.W, padx=16)
        self.header_id = tk.Label(
            self.header_info,
            text="ID: 24501152",
            font=("Segoe UI", 12),
            bg=SIDEBAR_GRADIENT_TOP,
            fg=TEXT_COLOR
        )
        self.header_id.pack(anchor=tk.W, padx=16)
        self.header_course = tk.Label(
            self.header_info,
            text="CMPE574 Biometrics\n(Spring 2024/2025)",
            font=("Segoe UI", 12),
            bg=SIDEBAR_GRADIENT_TOP,
            fg=TEXT_COLOR,
            justify=tk.LEFT
        )
        self.header_course.pack(anchor=tk.W, padx=16, pady=(0, 5))

        # App title
        self.title_label = tk.Label(self.sidebar_frame, text="Voice Gender Recognition", font=("Segoe UI", 20, "bold"), bg=SIDEBAR_GRADIENT_TOP, fg=ACCENT_GRADIENT, pady=18)
        self.title_label.pack()

        # Model selection with rounded frame
        self.model_frame = tk.LabelFrame(self.sidebar_frame, text="Model Selection", font=("Segoe UI", 14, "bold"), bg=SIDEBAR_GRADIENT_TOP, fg=TEXT_COLOR, relief=tk.GROOVE, bd=2, highlightbackground=ACCENT_GRADIENT, highlightcolor=ACCENT_GRADIENT)
        self.model_frame.pack(fill=tk.X, padx=20, pady=(10, 10))
        self.model_frame.config(borderwidth=2)
        self.use_svm = tk.BooleanVar(value=True)
        self.use_rf = tk.BooleanVar(value=True)
        self.use_cnn = tk.BooleanVar(value=True)
        self.svm_check = tk.Checkbutton(self.model_frame, text="SVM", variable=self.use_svm, font=("Segoe UI", 12), bg=SIDEBAR_GRADIENT_TOP, fg=TEXT_COLOR, selectcolor=ACCENT_GRADIENT, activebackground=SIDEBAR_GRADIENT_TOP, activeforeground=TEXT_COLOR, borderwidth=0, highlightthickness=0)
        self.svm_check.pack(anchor=tk.W, padx=10)
        self.rf_check = tk.Checkbutton(self.model_frame, text="Random Forest", variable=self.use_rf, font=("Segoe UI", 12), bg=SIDEBAR_GRADIENT_TOP, fg=TEXT_COLOR, selectcolor=ACCENT_GRADIENT, activebackground=SIDEBAR_GRADIENT_TOP, activeforeground=TEXT_COLOR, borderwidth=0, highlightthickness=0)
        self.rf_check.pack(anchor=tk.W, padx=10)
        self.cnn_check = tk.Checkbutton(self.model_frame, text="CNN", variable=self.use_cnn, font=("Segoe UI", 12), bg=SIDEBAR_GRADIENT_TOP, fg=TEXT_COLOR, selectcolor=ACCENT_GRADIENT, activebackground=SIDEBAR_GRADIENT_TOP, activeforeground=TEXT_COLOR, borderwidth=0, highlightthickness=0)
        self.cnn_check.pack(anchor=tk.W, padx=10)

        # Button style
        button_style = {
            "font": ("Segoe UI", 14, "bold"),
            "relief": tk.FLAT,
            "bd": 0,
            "cursor": "hand2",
            "height": 2,
            "activebackground": ACCENT_GRADIENT2,
            "activeforeground": "white"
        }
        # Train button
        self.train_button = tk.Button(self.sidebar_frame, text="Train Models", command=self.train_models_ui, bg=ACCENT_GRADIENT, fg="white", **button_style)
        self.train_button.pack(fill=tk.X, padx=30, pady=(10, 10))
        self.train_button.bind("<Enter>", lambda e: self.train_button.config(bg=ACCENT_GRADIENT2))
        self.train_button.bind("<Leave>", lambda e: self.train_button.config(bg=ACCENT_GRADIENT))
        # Predict button
        self.predict_button = tk.Button(self.sidebar_frame, text="Predict from Folder", command=self.predict_from_folder_ui, bg=ACCENT_GRADIENT, fg="white", **button_style)
        self.predict_button.pack(fill=tk.X, padx=30, pady=10)
        self.predict_button.bind("<Enter>", lambda e: self.predict_button.config(bg=ACCENT_GRADIENT2))
        self.predict_button.bind("<Leave>", lambda e: self.predict_button.config(bg=ACCENT_GRADIENT))
        # Show Analysis button with glow
        self.analysis_button = tk.Button(self.sidebar_frame, text="Show Analysis", command=self.show_analysis, bg=GLOW_COLOR, fg=SIDEBAR_GRADIENT_TOP, **button_style)
        self.analysis_button.pack(fill=tk.X, padx=30, pady=(10, 20))
        self.analysis_button.bind("<Enter>", lambda e: self.analysis_button.config(bg="#ffe9b0"))
        self.analysis_button.bind("<Leave>", lambda e: self.analysis_button.config(bg=GLOW_COLOR))
        self.analysis_button.config(highlightbackground=GLOW_COLOR, highlightcolor=GLOW_COLOR, highlightthickness=2)

        # Settings with rounded frame
        self.settings_frame = tk.LabelFrame(self.sidebar_frame, text="Settings", font=("Segoe UI", 14, "bold"), bg=SIDEBAR_GRADIENT_TOP, fg=TEXT_COLOR, relief=tk.GROOVE, bd=2, highlightbackground=ACCENT_GRADIENT, highlightcolor=ACCENT_GRADIENT)
        self.settings_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.settings_frame.config(borderwidth=2)
        self.train_audio_dir_frame = tk.Frame(self.settings_frame, bg=SIDEBAR_GRADIENT_TOP)
        self.train_audio_dir_label = tk.Label(self.train_audio_dir_frame, text="Post-training test dir:", font=("Segoe UI", 12), bg=SIDEBAR_GRADIENT_TOP, fg=TEXT_COLOR)
        self.train_audio_dir_label.pack(side=tk.LEFT, padx=5)
        self.train_audio_dir_var = tk.StringVar(value="audio")
        self.train_audio_dir_entry = tk.Entry(self.train_audio_dir_frame, textvariable=self.train_audio_dir_var, width=18, font=("Segoe UI", 12), relief=tk.GROOVE, bd=2)
        self.train_audio_dir_entry.pack(side=tk.LEFT, padx=5)
        self.train_audio_dir_browse_button = tk.Button(self.train_audio_dir_frame, text="Browse", command=lambda: self.browse_directory(self.train_audio_dir_var), font=("Segoe UI", 11), bg=ACCENT_GRADIENT, fg="white", relief=tk.FLAT, cursor="hand2", bd=0, activebackground=ACCENT_GRADIENT2)
        self.train_audio_dir_browse_button.pack(side=tk.LEFT, padx=5)
        self.train_audio_dir_browse_button.bind("<Enter>", lambda e: self.train_audio_dir_browse_button.config(bg=ACCENT_GRADIENT2))
        self.train_audio_dir_browse_button.bind("<Leave>", lambda e: self.train_audio_dir_browse_button.config(bg=ACCENT_GRADIENT))
        self.train_audio_dir_frame.pack(fill=tk.X, pady=5)
        self.predict_audio_dir_frame = tk.Frame(self.settings_frame, bg=SIDEBAR_GRADIENT_TOP)
        self.predict_audio_dir_label = tk.Label(self.predict_audio_dir_frame, text="Prediction dir:", font=("Segoe UI", 12), bg=SIDEBAR_GRADIENT_TOP, fg=TEXT_COLOR)
        self.predict_audio_dir_label.pack(side=tk.LEFT, padx=5)
        self.predict_audio_dir_var = tk.StringVar(value="audio")
        self.predict_audio_dir_entry = tk.Entry(self.predict_audio_dir_frame, textvariable=self.predict_audio_dir_var, width=18, font=("Segoe UI", 12), relief=tk.GROOVE, bd=2)
        self.predict_audio_dir_entry.pack(side=tk.LEFT, padx=5)
        self.predict_audio_dir_browse_button = tk.Button(self.predict_audio_dir_frame, text="Browse", command=lambda: self.browse_directory(self.predict_audio_dir_var), font=("Segoe UI", 11), bg=ACCENT_GRADIENT, fg="white", relief=tk.FLAT, cursor="hand2", bd=0, activebackground=ACCENT_GRADIENT2)
        self.predict_audio_dir_browse_button.pack(side=tk.LEFT, padx=5)
        self.predict_audio_dir_browse_button.bind("<Enter>", lambda e: self.predict_audio_dir_browse_button.config(bg=ACCENT_GRADIENT2))
        self.predict_audio_dir_browse_button.bind("<Leave>", lambda e: self.predict_audio_dir_browse_button.config(bg=ACCENT_GRADIENT))
        self.predict_audio_dir_frame.pack(fill=tk.X, pady=5)

        # Status
        self.status_label = tk.Label(self.sidebar_frame, text="Status: Ready", font=("Segoe UI", 12), bg=SIDEBAR_GRADIENT_TOP, fg=TEXT_COLOR, anchor=tk.W, pady=10)
        self.status_label.pack(fill=tk.X, padx=20, pady=(10, 0))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.sidebar_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.run_action_button = tk.Button(
            self.sidebar_frame, text="Run Action", command=self.run_action,
            font=("Segoe UI", 14, "bold"), bg=SUCCESS_COLOR, fg="white",
            relief=tk.FLAT, pady=10, cursor="hand2", bd=0, activebackground="#27ae60"
        )
        self.run_action_button.pack(fill=tk.X, padx=30, pady=(0, 20))
        self.run_action_button.bind("<Enter>", lambda e: self.run_action_button.config(bg="#27ae60"))
        self.run_action_button.bind("<Leave>", lambda e: self.run_action_button.config(bg=SUCCESS_COLOR))

        # Main area with tabs (responsive, card style)
        self.main_area = ttk.Notebook(master)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook.Tab', background="#232946", foreground=HEADER_COLOR, font=("Segoe UI", 15, "bold"), padding=[20, 10])
        style.configure('TNotebook', background=DARK_BG, borderwidth=0)

        # Results tab with card style
        self.results_tab = tk.Frame(self.main_area, bg=DARK_BG)
        self.main_area.add(self.results_tab, text="Results & Logs")
        self.results_text_frame = tk.Frame(self.results_tab, bg=DARK_BG)
        self.results_text_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        self.results_card = tk.Frame(self.results_text_frame, bg="#232946", bd=0, relief=tk.RIDGE, highlightbackground=ACCENT_GRADIENT, highlightcolor=ACCENT_GRADIENT, highlightthickness=2)
        self.results_card.pack(fill=tk.BOTH, expand=True)
        self.results_text_scroll = tk.Scrollbar(self.results_card)
        self.results_text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text = tk.Text(self.results_card, wrap=tk.WORD, font=("Consolas", 13), bg="#232946", fg=TEXT_COLOR, height=20, borderwidth=0, relief=tk.FLAT, yscrollcommand=self.results_text_scroll.set)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.results_text_scroll.config(command=self.results_text.yview)
        self.results_text.configure(state='disabled')
        self.results_redirector = ResultsRedirector(self.results_text)

        # Figures tab with card style
        self.figures_tab = tk.Frame(self.main_area, bg=DARK_BG)
        self.main_area.add(self.figures_tab, text="Analysis Figures")
        self.figure_names = [

            ("Feature Correlation Heatmap", "figures/feature_correlation_heatmap.png")
        ]
        self.current_figure_idx = 0
        self.figure_panel = tk.Frame(self.figures_tab, bg=DARK_BG)
        self.figure_panel.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        self.figure_card = tk.Frame(self.figure_panel, bg="#232946", bd=0, relief=tk.RIDGE, highlightbackground=ACCENT_GRADIENT, highlightcolor=ACCENT_GRADIENT, highlightthickness=2)
        self.figure_card.pack(fill=tk.BOTH, expand=True)
        self.figure_title = tk.Label(self.figure_card, text="", font=("Segoe UI", 16, "bold"), bg="#232946", fg=ACCENT_GRADIENT)
        self.figure_title.pack(pady=(20, 10))
        self.figure_canvas_frame = tk.Frame(self.figure_card, bg="#232946")
        self.figure_canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.figure_canvas_scroll = tk.Scrollbar(self.figure_canvas_frame, orient=tk.VERTICAL)
        self.figure_canvas_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.figure_canvas = tk.Label(self.figure_canvas_frame, bg="#232946")
        self.figure_canvas.pack(fill=tk.BOTH, expand=True)
        self.figure_canvas_scroll.config(command=self._on_figure_scroll)
        nav_frame = tk.Frame(self.figure_card, bg="#232946")
        nav_frame.pack(pady=20)
        self.prev_btn = tk.Button(nav_frame, text="◀ Prev", command=self.show_prev_figure, font=("Segoe UI", 13, "bold"), bg=ACCENT_GRADIENT, fg="white", relief=tk.FLAT, cursor="hand2", bd=0, activebackground=ACCENT_GRADIENT2)
        self.prev_btn.pack(side=tk.LEFT, padx=10)
        self.prev_btn.bind("<Enter>", lambda e: self.prev_btn.config(bg=ACCENT_GRADIENT2))
        self.prev_btn.bind("<Leave>", lambda e: self.prev_btn.config(bg=ACCENT_GRADIENT))
        self.next_btn = tk.Button(nav_frame, text="Next ▶", command=self.show_next_figure, font=("Segoe UI", 13, "bold"), bg=ACCENT_GRADIENT, fg="white", relief=tk.FLAT, cursor="hand2", bd=0, activebackground=ACCENT_GRADIENT2)
        self.next_btn.pack(side=tk.LEFT, padx=10)
        self.next_btn.bind("<Enter>", lambda e: self.next_btn.config(bg=ACCENT_GRADIENT2))
        self.next_btn.bind("<Leave>", lambda e: self.next_btn.config(bg=ACCENT_GRADIENT))
        self.update_figure_panel()

        self.current_action = None
        self.check_existing_models()

    def check_existing_models(self):
        model_dir = "saved_models"
        if os.path.exists(model_dir):
            has_svm = os.path.exists(os.path.join(model_dir, "svm_model.joblib"))
            has_rf = os.path.exists(os.path.join(model_dir, "rf_model.joblib"))
            has_cnn = os.path.exists(os.path.join(model_dir, "cnn_model.pth"))
            self.use_svm.set(has_svm)
            self.use_rf.set(has_rf)
            self.use_cnn.set(has_cnn)
            if has_svm or has_rf or has_cnn:
                self.status_label.config(text="Found existing models. Ready to predict.")

    def browse_directory(self, dir_var):
        directory = filedialog.askdirectory()
        if directory:
            dir_var.set(directory)

    def train_models_ui(self):
        self.current_action = "train"
        self.status_label.config(text="Ready to train. Specify optional audio dir for post-training eval.")
        self.progress_var.set(0)
        self.clear_results()
        self.main_area.select(self.results_tab)

    def predict_from_folder_ui(self):
        self.current_action = "predict"
        self.status_label.config(text="Ready to predict. Specify audio directory.")
        self.progress_var.set(0)
        self.clear_results()
        self.main_area.select(self.results_tab)

    def clear_results(self):
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.configure(state='disabled')

    def run_action(self):
        if self.current_action == "train":
            audio_dir = self.train_audio_dir_var.get()
            if not audio_dir:
                audio_dir = "audio"
            if not os.path.isdir(audio_dir) and audio_dir != "audio":
                messagebox.showerror("Error", f"Post-training audio directory '{audio_dir}' not found.")
                return
            self.status_label.config(text=f"Training models... (Post-eval dir: {audio_dir})")
            self.progress_var.set(10)
            self.clear_results()
            threading.Thread(target=self.execute_training, args=(audio_dir,), daemon=True).start()
        elif self.current_action == "predict":
            audio_dir = self.predict_audio_dir_var.get()
            if not audio_dir or not os.path.isdir(audio_dir):
                messagebox.showerror("Error", f"Audio directory '{audio_dir}' not found or not specified.")
                return
            if not (self.use_svm.get() or self.use_rf.get() or self.use_cnn.get()):
                messagebox.showerror("Error", "Please select at least one model to use for prediction.")
                return
            self.status_label.config(text=f"Predicting from '{audio_dir}'...")
            self.progress_var.set(10)
            self.clear_results()
            threading.Thread(target=self.execute_prediction, args=(audio_dir,), daemon=True).start()
        else:
            messagebox.showinfo("Info", "Please select an action (Train or Predict) first.")

    def execute_training(self, audio_dir):
        try:
            print(f"Starting training with post-evaluation on: {audio_dir}")
            self.progress_var.set(20)
            perform_training_and_evaluation(audio_dir_for_post_training_eval=audio_dir)
            self.progress_var.set(100)
            self.master.after(0, lambda: self.status_label.config(text="Training completed!"))
            self.master.after(0, lambda: messagebox.showinfo("Success", "Model training and evaluation finished."))
            self.check_existing_models()
        except Exception as e:
            self.progress_var.set(0)
            self.master.after(0, lambda: self.status_label.config(text="Error during training."))
            self.master.after(0, lambda: messagebox.showerror("Training Error", str(e)))

    def execute_prediction(self, audio_dir):
        try:
            print(f"Starting prediction on: {audio_dir}")
            self.progress_var.set(20)
            perform_prediction_with_saved_models(audio_dir_to_predict=audio_dir)
            self.progress_var.set(100)
            self.master.after(0, lambda: self.status_label.config(text="Prediction completed!"))
            self.master.after(0, lambda: messagebox.showinfo("Success", f"Prediction on '{audio_dir}' finished."))
        except Exception as e:
            self.progress_var.set(0)
            self.master.after(0, lambda: self.status_label.config(text="Error during prediction."))
            self.master.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))

    # --- Analysis Figures ---
    def show_analysis(self):
        self.status_label.config(text="Generating analysis figures...")
        self.progress_var.set(10)
        threading.Thread(target=self._generate_and_show_figures, daemon=True).start()

    def _generate_and_show_figures(self):
        try:
            generate_analysis_figures()
            self.progress_var.set(100)
            self.master.after(0, lambda: self.status_label.config(text="Analysis figures generated!"))
            self.master.after(0, self.update_figure_panel)
            self.master.after(0, lambda: self.main_area.select(self.figures_tab))
        except Exception as e:
            self.progress_var.set(0)
            self.master.after(0, lambda: self.status_label.config(text="Error generating analysis figures."))
            self.master.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))

    def update_figure_panel(self):
        name, path = self.figure_names[self.current_figure_idx]
        self.figure_title.config(text=name)
        if os.path.exists(path):
            img = Image.open(path)
            img = img.resize((700, 450), Image.LANCZOS)
            self.figure_img = ImageTk.PhotoImage(img)
            self.figure_canvas.config(image=self.figure_img)
        else:
            self.figure_canvas.config(image='', text='Figure not found', fg=WARNING_COLOR)

    def show_prev_figure(self):
        self.current_figure_idx = (self.current_figure_idx - 1) % len(self.figure_names)
        self.update_figure_panel()

    def show_next_figure(self):
        self.current_figure_idx = (self.current_figure_idx + 1) % len(self.figure_names)
        self.update_figure_panel()

    def _on_figure_scroll(self, *args):
        # Placeholder for future scroll logic if needed
        pass

if __name__ == '__main__':
    root = tk.Tk()
    app = VoiceGenderApp(root)
    root.mainloop()
