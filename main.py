import librosa
import librosa.display
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from playsound import playsound
import soundfile as sf

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Audio - Bandas Sonoras")
        self.root.geometry("800x600")

        # Variables para audio
        self.y = None
        self.sr = None

        # Botón para cargar archivo de audio
        load_button = tk.Button(self.root, text="Cargar archivo de audio", command=self.load_audio)
        load_button.pack(pady=10)

        # Botón para reproducir audio
        self.play_button = tk.Button(self.root, text="Reproducir Audio", command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(pady=10)

        # Botón para analizar audio
        self.analyze_button = tk.Button(self.root, text="Analizar Audio", command=self.analyze_audio, state=tk.DISABLED)
        self.analyze_button.pack(pady=10)

        # Etiqueta para mostrar información del archivo
        self.info_label = tk.Label(self.root, text="No se ha cargado ningún archivo", font=("Arial", 12))
        self.info_label.pack(pady=10)

        # Canvas para mostrar el espectrograma y gráficos
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de audio", "*.wav")])
        if not file_path:
            return

        try:
            self.y, self.sr = sf.read(file_path)
            duration = len(self.y) / self.sr
            channels = "Mono" if len(self.y.shape) == 1 else "Estéreo"

            # Actualizar información
            self.info_label.config(
                text=f"Archivo: {file_path}\nDuración: {duration:.2f} segundos\nFrecuencia de muestreo: {self.sr} Hz\nCanales: {channels}"
            )

            # Habilitar botones
            self.play_button.config(state=tk.NORMAL)
            self.analyze_button.config(state=tk.NORMAL)

            # Mostrar espectrograma
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            self.plot_spectrogram(self.y, self.sr)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo de audio.\nError: {e}")

    def plot_spectrogram(self, y, sr):
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(8, 6))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
        ax.set(title="Espectrograma Mel")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def play_audio(self):
        if self.y is not None:
            try:
                temp_file = "temp_audio.wav"
                sf.write(temp_file, self.y, self.sr)
                playsound(temp_file)
            except Exception as e:
                messagebox.showerror("Error", f"Error al reproducir el audio: {e}")

    def analyze_audio(self):
        if self.y is not None:
            try:
                if len(self.y.shape) > 1:
                    y_mono = np.mean(self.y, axis=1)
                else:
                    y_mono = self.y

                # Calcular tempo y beats
                tempo, beats = librosa.beat.beat_track(y=y_mono, sr=self.sr)
                beat_times = librosa.frames_to_time(beats, sr=self.sr)

                # Mostrar resultados en mensaje (corregido)
                messagebox.showinfo(
                    "Análisis del Audio",
                    f"Tempo: {tempo:.2f} BPM\nNúmero de beats detectados: {len(beats)}\n"
                    f"Primera beat: {beat_times[0]:.2f} segundos" if len(beat_times) > 0 else "No se detectaron beats"
                )

            except Exception as e:
                messagebox.showerror("Error", f"Error durante el análisis del audio: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()