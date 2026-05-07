import tkinter as tk
from realtime_prediction import start_realtime_prediction

root = tk.Tk()

root.title("Sign Language To Text")

root.geometry("400x300")

title = tk.Label(root,text="Sign Language Recognition",
                 font=("Arial",18))

title.pack(pady=20)

start_btn = tk.Button(root,
                      text="Start Camera",
                      font=("Arial",14),
                      command=start)

start_btn.pack(pady=20)

exit_btn = tk.Button(root,
                     text="Exit",
                     font=("Arial",14),
                     command=root.destroy)

exit_btn.pack(pady=20)

root.mainloop()