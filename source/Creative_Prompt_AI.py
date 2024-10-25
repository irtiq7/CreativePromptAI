import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import subprocess
import sys
import os
import threading

# Global flag to control thread termination
app_running = True

# Function to update the loading bar with a percentage
def update_loading_bar(progress, total_steps):
    percentage = int((progress / total_steps) * 100)
    loading_bar['value'] = percentage
    loading_label.config(text=f"{percentage}%")
    root.update_idletasks()

# Function to display thumbnails of generated images in a grid with multiple columns
def display_thumbnails(images):
    from PIL import Image, ImageTk  # Ensure PIL is imported after installation
    # Clear previous thumbnails
    for widget in right_frame.winfo_children():
        widget.destroy()

    num_columns = 5  # Set the number of columns to 5 (or any desired number)
    
    for i, img in enumerate(images):
        img_thumbnail = img.copy().resize((100, 100), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_thumbnail)
        
        # Calculate the row and column based on the index and the number of columns
        row = i // num_columns
        col = i % num_columns

        # Place thumbnails in a grid without too much padding
        thumbnail_button = tk.Button(right_frame, image=img_tk, command=lambda img=img: display_image(img))
        thumbnail_button.image = img_tk  # Keep a reference to avoid garbage collection
        thumbnail_button.grid(row=row, column=col, padx=2, pady=2)  # Reduce padx and pady to bring them closer

# Function to display image in a new window with zoom option
def display_image(image):
    from PIL import Image, ImageTk  # Ensure PIL is imported after installation
    window = tk.Toplevel()
    window.title("Image Viewer")

    img = ImageTk.PhotoImage(image)
    label = tk.Label(window, image=img)
    label.image = img  # Keep a reference to avoid garbage collection
    label.pack()

    # Save button
    def save_image():
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            image.save(file_path)

    save_button = tk.Button(window, text="Save", command=save_image)
    save_button.pack()

    # Zoom in and out
    def zoom(event):
        if event.delta > 0:
            new_size = (int(image.width * 1.1), int(image.height * 1.1))
        else:
            new_size = (int(image.width * 0.9), int(image.height * 0.9))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(resized_image)
        label.config(image=img)
        label.image = img

    window.bind("<MouseWheel>", zoom)

# Function to handle application close event
def on_closing():
    global app_running
    app_running = False  # Signal threads to stop
    root.quit()  # Stop the mainloop
    root.destroy()  # Close the application

# Function to generate images and show thumbnails with progress
def generate_images():
    global images
    try:
        from diffusers import DiffusionPipeline, AutoPipelineForText2Image
        import torch
        from PIL import Image, ImageTk  # Ensure PIL is imported after installation

        num_images = int(num_images_entry.get())
    except ValueError:
        response_text.set("Invalid number of images. Please enter a valid number.")
        return

    images = []
    prompt = prompt_text.get("1.0", "end-1c")  # Get multi-line input from Text widget
    negative_prompt = "3d render, realistic"
    
    response_text.set(f"Generating {num_images} images...")
    root.update_idletasks()

    # Reset progress bar
    loading_bar['value'] = 0
    loading_label.config(text="0%")

    # Total steps is equal to the number of images
    total_steps = num_images

    for i in range(num_images):
        if not app_running:
            break  # If app is closing, stop generating images

        img = pipe(
            prompt=prompt
        ).images[0]
        
        img.save(f"image_{i}.png")
        images.append(img)

        # Update progress bar
        update_loading_bar(i + 1, total_steps)

    display_thumbnails(images)
    response_text.set(f"Generated {num_images} images successfully.")
    loading_label.config(text="100%")

# Function to install dependencies with progress bar and real-time updates
def install_dependencies():
    try:
        total_steps = 6  # Adjusted total steps to match the number of installations
        response_text.set("Installing dependencies...")

        # Step 1: Install diffusers
        update_loading_bar(1, total_steps)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "diffusers"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)

        # Step 2: Install torch with CUDA support
        update_loading_bar(2, total_steps)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)

        # Step 3: Install Pillow
        update_loading_bar(3, total_steps)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "Pillow"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)

        # Step 4: Install llama-cpp-python
        update_loading_bar(4, total_steps)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "llama-cpp-python==0.2.77", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)

        # Step 5: Install transformers
        update_loading_bar(5, total_steps)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "transformers"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)

        # Step 6: Install accelerate
        update_loading_bar(6, total_steps)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "accelerate"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        # Step 7: Install PEFT
        update_loading_bar(7, total_steps)
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "peft"], check=True, creationflags=subprocess.CREATE_NO_WINDOW)

        response_text.set("Dependencies installed successfully.")
        loading_bar['value'] = 100
        loading_label.config(text="100%")
    except subprocess.CalledProcessError as e:
        response_text.set(f"Error installing dependencies: {e}")
    except Exception as e:
        response_text.set(f"Unexpected error: {e}")

# Function to load the model with manual progress handling
def load_model():
    global pipe
    try:
        from diffusers import DiffusionPipeline, AutoPipelineForText2Image
        import torch

        model_id = model_id_entry.get()
        lora_weight = lora_weight_entry.get()
        download_path = download_path_entry.get()
        response_text.set("Loading model...")
        root.update_idletasks()

        # Total steps for loading pipeline components
        total_steps = 8

        # Reset progress bar
        loading_bar['value'] = 0

        # Simulate model loading in steps
        # Step 1: Load model components
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            cache_dir=download_path
        )
        update_loading_bar(2, total_steps)

        # Step 2: Move model to GPU
        pipe.to(device="cuda", dtype=torch.float16)
        update_loading_bar(3, total_steps)

        # Step 3: Load LoRA weights
        if not len(lora_weight) == 0:
            pipe.load_lora_weights(lora_weight)
        update_loading_bar(8, total_steps)

        # Final update to show complete
        response_text.set("Model loaded successfully.")
        loading_bar['value'] = 100
        loading_label.config(text="100%")
    except ImportError as e:
        response_text.set(f"Module not found: {e}. Please install dependencies.")
    except Exception as e:
        response_text.set(f"Error loading model: {e}")
        
# Function to display the About dialog box
def show_about_dialog():
    about_info = "Creator: Usama Saqib\nGitHub: irtiq7\nInfo: Creative Prompt AI is a powerful open-source desktop application designed to make AI image generation and viewing seamless and intuitive."
    messagebox.showinfo("About Creative Prompt AI", about_info)

# Main GUI window
root = tk.Tk()
root.title("Creative Prompt AI")
root.geometry("800x600")
root.resizable(True, True)

# PanedWindow for resizable panes
paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
paned_window.pack(fill=tk.BOTH, expand=True)

# Left frame for controls
left_frame = tk.Frame(paned_window)
paned_window.add(left_frame)

# Right frame for image viewer
right_frame = tk.Frame(paned_window, borderwidth=4, relief="solid")
paned_window.add(right_frame)

# Configure grid layout for left frame
left_frame.grid_columnconfigure(0, weight=1)

# Model ID label and entry
model_id_label = tk.Label(left_frame, text="Model ID:")
model_id_label.grid(row=0, column=0, sticky="w", padx=5, pady=(10, 0))  # Padding added for space above the text box
model_id_entry = tk.Entry(left_frame)
model_id_entry.insert(0, "stabilityai/stable-diffusion-xl-base-1.0")
model_id_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

# Lora Weights label and entry
lora_weight_label = tk.Label(left_frame, text="Lora Weight (Optional):")
lora_weight_label.grid(row=2, column=0, sticky="w", padx=5, pady=(10, 0))  # Padding added for space above the text box
lora_weight_entry = tk.Entry(left_frame)
lora_weight_entry.insert(0, "nerijs/pixel-art-xl")
lora_weight_entry.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

# Download path label and entry
download_path_label = tk.Label(left_frame, text="Download Path:")
download_path_label.grid(row=4, column=0, sticky="w", padx=5, pady=(10, 0))
download_path_entry = tk.Entry(left_frame)
download_path_entry.insert(0, os.getcwd())
download_path_entry.grid(row=5, column=0, sticky="ew", padx=5, pady=5)

# Prompt label and entry (using a Text widget for multi-line input)
prompt_label = tk.Label(left_frame, text="Prompt:")
prompt_label.grid(row=6, column=0, sticky="w", padx=5, pady=(10, 0))
prompt_text = tk.Text(left_frame, height=5)
prompt_text.insert("1.0", "pixel art, a cute corgi, simple, flat colors")
prompt_text.grid(row=7, column=0, sticky="ew", padx=5, pady=5)

# Number of images label and entry
num_images_label = tk.Label(left_frame, text="Number of Images:")
num_images_label.grid(row=8, column=0, sticky="w", padx=5, pady=(10, 0))
num_images_entry = tk.Entry(left_frame)
num_images_entry.insert(0, "9")
num_images_entry.grid(row=9, column=0, sticky="ew", padx=5, pady=5)

# Load model button
load_model_button = tk.Button(left_frame, text="Load Model", command=lambda: threading.Thread(target=load_model).start())
load_model_button.grid(row=10, column=0, pady=5, sticky="ew")

# Response text box
response_text = tk.StringVar()
response_label = tk.Label(left_frame, textvariable=response_text)
response_label.grid(row=14, column=0, pady=5)

# Install dependencies button
install_button = tk.Button(left_frame, text="Install Dependencies", command=lambda: threading.Thread(target=install_dependencies).start())
install_button.grid(row=11, column=0, pady=5, sticky="ew")

# Generate images button
generate_button = tk.Button(left_frame, text="Generate Images", command=lambda: threading.Thread(target=generate_images).start())
generate_button.grid(row=12, column=0, pady=5, sticky="ew")

# Loading bar and percentage label
loading_bar = Progressbar(left_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
loading_bar.grid(row=599, column=0, pady=10, sticky="ew")
loading_label = tk.Label(left_frame, text="0%")
loading_label.grid(row=600, column=0)

# About button
about_button = tk.Button(left_frame, text="About", command=show_about_dialog)
about_button.grid(row=13, column=0, pady=10, sticky="ew")

# Bind the close event to on_closing
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the main loop
root.mainloop()
