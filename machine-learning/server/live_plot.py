import pyaudio
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Global variables
audio_data = np.zeros(CHUNK)
fig, ax = plt.subplots()
line, = ax.plot(audio_data)
ax.set_ylim([-32768, 32767])  # Adjust the y-axis limits based on your audio data range

# Create the audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Create the GUI
root = tk.Tk()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Function to update the GUI with the waveform
def update_gui():
    while True:
        line.set_ydata(audio_data)
        canvas.draw()
        canvas.flush_events()

# Create a thread for updating the GUI
update_thread = threading.Thread(target=update_gui)
update_thread.start()

# Function to record audio
def record_audio():
    global audio_data
    while True:
        try:
            # Read audio data from the stream
            in_data = stream.read(CHUNK)
            audio_data = np.frombuffer(in_data, dtype=np.int16)

            # You can add additional processing logic here

        except KeyboardInterrupt:
            break

# Create a thread for recording audio
record_thread = threading.Thread(target=record_audio)
record_thread.start()

# Run the GUI
root.mainloop()

# Stop and close the stream when the GUI is closed
stream.stop_stream()
stream.close()
p.terminate()
