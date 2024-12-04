import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

from tkinter import filedialog
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class analizadorAudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis de Banda Sonora")
        self.root.geometry("800x600")

        # Boton para poder cargar el archivo
        btn_cargar = tk.Button(self.root, text="Cargar archivo de audio (.wav)", command=self.cargar_audio)
        btn_cargar.pack(pady=10)

        # Canvas para mostrar el espectograma (pa apantallar)
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def cargar_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de audio", "*.wav")])
        if not file_path:
            return

        try:
            # Cargar archivo de audio
            y, sr = librosa.load(file_path, sr=None)
            messagebox.showinfo("Archivo cargado", f"Archivo cargado correctamente: {file_path}")

            # Limpiamos el canvas anterior
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            # Mostramos el espectrograma
            self.plot_espectrograma(y, sr)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar archivo audio.\nError: {e}")

    def plot_espectrograma(self, y, sr):
        # Nos aventamos el calculo del espectrograma
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Creamos la figura del espectrograma
        fig, ax = plt.subplots(figsize=(8, 6))
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
        ax.set(title="Espectrograma Mel")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        # Colocamos el grafico en la interfaz
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = analizadorAudio(root)
    root.mainloop()