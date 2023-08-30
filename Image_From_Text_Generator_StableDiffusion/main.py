# Libraries for GUI
import tkinter as tk
import customtkinter as ctk


# Libraries for ML
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import os


# Libraries for Processing Image
from PIL import ImageTk


# Private modules
import auth_token


# Creating User Interface
app = tk.Tk()
app.geometry('532x632')
app.title('Image Generator')
app.configure(bg= 'black')
ctk.set_appearance_mode('dark')


# Creating Input Box
prompt = ctk.CTkEntry(height= 40,
                      width= 512,
                      text_font= ('Arial',  16),
                      text_color= 'black',
                      fg_color= 'white'
                      )
prompt.place(x=10, y=10)


# PlaceHolder for Generated Image
img_placeholder = ctk.CTkLabel(height= 512,
                               width= 512,
                               text=''
                               )

img_placeholder.place(x= 10,
                      y= 110
                      )


# Download Stable Diffusion Model
modelid = "CompVis/stable-diffusion-v1-4"
device = 'cuda'

stable_diffusion_model = StableDiffusionPipeline.from_pretrained(modelid,
                                                                 revision= 'fp16',
                                                                 torch_dtype= torch.float32,
                                                                 use_auth_token = auth_token
                                                                 )


# Generating Image From Text
def generate_image():
    with autocast(device):
        image = stable_diffusion_model(prompt.get(), guidance_scale= 8.5).images[0]


    # Saving Image
    image.save('generated_img.png')


    # Display Image
    img = ImageTk.PhotoImage(image)
    img_placeholder.configure(image= img)


# Creating a Button for the 'generate_image function'
trigger = ctk.CTkButton(height= 40,
                        width= 120,
                        text_font= ('Arial', 15),
                        text_color= 'black',
                        fg_color= 'white',
                        command= generate_image
                        )

trigger.configure(text= 'Generate')
trigger.place(x= 206,
              y= 60
              )

app.mainloop()