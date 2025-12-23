import customtkinter as ctk
from tkinter import messagebox
import mss
import numpy as np
import cv2



import customtkinter as ctk

from classprocessor import Processor,start_detection   # <-- your other file

class CameraApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Camera Control")
        self.geometry("400x300")

        self.processor = Processor()

        self.main_menu = MainMenu(self)
        self.camera_screen = CameraScreen(self)

        self.show_frame(self.main_menu)

    def show_frame(self, frame):
        for widget in self.winfo_children():
            widget.pack_forget()
        frame.pack(expand=True, fill="both")


# -------------------------------
# MAIN MENU
# -------------------------------
class MainMenu(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        ctk.CTkLabel(self, text="Camera Options", font=("Arial", 20)).pack(pady=30)

        ctk.CTkButton(
            self,
            text="Register Camera",
            command=self.register_camera
        ).pack(pady=10)

        ctk.CTkButton(
            self,
            text="Use Camera",
            command=self.use_camera
        ).pack(pady=10)

    def register_camera(self):
        self.master.show_frame(self.master.camera_screen)
        self.master.camera_screen.set_mode("register")
        self.master.processor.start_segmentation()

    def use_camera(self):
        self.master.show_frame(self.master.camera_screen)
        self.master.camera_screen.set_mode("use")
        start_detection(self.master.processor)


# -------------------------------
# CAMERA SCREEN
# -------------------------------
class CameraScreen(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.label = ctk.CTkLabel(self, text="", font=("Arial", 18))
        self.label.pack(pady=20)

        ctk.CTkButton(
            self,
            text="Go Back",
            command=self.go_back
        ).pack(pady=20)

    def set_mode(self, mode):
        self.label.configure(
            text="Registering Camera..." if mode == "register" else "Using Camera..."
        )

    def go_back(self):
        #self.master.processor.stop()  #inconsistent
        self.master.show_frame(self.master.main_menu)


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = CameraApp()
    app.mainloop()
