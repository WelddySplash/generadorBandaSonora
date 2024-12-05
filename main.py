import librosa
import librosa.display
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os
import subprocess
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador de Composiciones Musicales")
        self.root.geometry("800x600")

        # Variables de audio
        self.y = None
        self.sr = None
        self.tempo = None
        self.model = None

        # Frame para los botones
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=0, column=0, pady=10, padx=10)

        # Botón para cargar audio
        load_button = tk.Button(button_frame, text="Cargar Audio", command=self.load_audio)
        load_button.grid(row=0, column=0, padx=5)

        # Botón para analizar audio
        analyze_button = tk.Button(button_frame, text="Analizar Audio", command=self.analyze_audio)
        analyze_button.grid(row=0, column=1, padx=5)

        # Botón para entrenar el modelo
        train_button = tk.Button(button_frame, text="Entrenar Modelo", command=self.train_model)
        train_button.grid(row=0, column=2, padx=5)

        # Botón para generar composición
        generate_button = tk.Button(button_frame, text="Generar Composición", command=self.generate_music)
        generate_button.grid(row=0, column=3, padx=5)

        # Botón para reproducir composición generada
        self.play_generated_button = tk.Button(button_frame, text="Reproducir Composición", command=self.play_generated_music, state=tk.DISABLED)
        self.play_generated_button.grid(row=0, column=4, padx=5)

        # Canvas para mostrar espectrograma
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.grid(row=2, column=0, sticky="nsew")
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de audio", "*.wav *.mp3")])
        if not file_path:
            return

        try:
            self.y, self.sr = librosa.load(file_path, sr=22050)

            # Validar si el archivo de audio contiene datos
            if self.y is None or len(self.y) == 0:
                raise ValueError("El archivo de audio está vacío o no contiene datos válidos.")

            self.plot_spectrogram(self.y, self.sr)
            messagebox.showinfo("Audio Cargado", f"Archivo cargado: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo de audio: {e}")

    def plot_spectrogram(self, y, sr):
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(8, 6))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set(title="Espectrograma Mel")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def analyze_audio(self):
        if self.y is not None and len(self.y) > 0:
            try:
                if len(self.y.shape) > 1:
                    y_mono = np.mean(self.y, axis=1)
                else:
                    y_mono = self.y

                self.tempo, beats = librosa.beat.beat_track(y=y_mono, sr=self.sr)
                self.tempo = self.tempo.item()
                beat_times = librosa.frames_to_time(beats, sr=self.sr)

                if len(beat_times) > 0:
                    messagebox.showinfo("Análisis del Audio", f"Tempo: {self.tempo:.2f} BPM\nNúmero de beats detectados: {len(beats)}")
                else:
                    messagebox.showwarning("Análisis del Audio", "No se detectaron beats en el audio.")
            except Exception as e:
                messagebox.showerror("Error", f"Error durante el análisis del audio: {e}")
        else:
            messagebox.showerror("Error", "El archivo de audio está vacío o no contiene datos válidos.")

    def train_model(self):
        input_folder = filedialog.askdirectory(title="Selecciona la carpeta con archivos de audio")
        if not input_folder or not os.listdir(input_folder):
            messagebox.showerror("Error", "La carpeta seleccionada está vacía o no existe.")
            return

        data = []
        for file_name in os.listdir(input_folder):
            if file_name.lower().endswith(".wav") or file_name.lower().endswith(".mp3"):
                try:
                    file_path = os.path.join(input_folder, file_name)
                    y, sr = librosa.load(file_path, sr=22050)
                    if len(y) > 0:
                        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
                        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
                        data.append(mfcc.T)
                    else:
                        print(f"El archivo {file_name} está vacío o no contiene audio válido.")
                except Exception as e:
                    print(f"Error al cargar el archivo {file_name}: {e}")

        if len(data) == 0:
            messagebox.showerror("Error", "No se encontraron archivos de audio válidos.")
            return

        max_length = max(m.shape[0] for m in data)
        data = pad_sequences(data, maxlen=max_length, padding='post', dtype='float32')

        X = np.array(data)

        self.model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),
            LSTM(256, return_sequences=True),
            Dense(X.shape[2], activation='linear')
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.model.fit(X, X, epochs=20, batch_size=16)
        messagebox.showinfo("Modelo Entrenado", "Modelo entrenado con éxito.")

    def generate_music(self):
        if not self.model:
            messagebox.showerror("Error", "Primero entrena el modelo.")
            return

        try:
            seed = np.expand_dims(librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=20, hop_length=512).T, axis=0)
            print("Seed shape before padding:", seed.shape)

            seed = pad_sequences(seed, maxlen=self.model.input_shape[1], padding='post', dtype='float32')
            print("Seed shape after padding:", seed.shape)

            output = self.model.predict(seed)
            print("Output shape:", output.shape)

            mel_spec = librosa.feature.inverse.mfcc_to_mel(output.T, n_mels=128, dct_type=2)
            print("Mel spectrogram shape:", mel_spec.shape)

            output_audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=self.sr, n_iter=64, hop_length=512)
            print("Output audio length:", len(output_audio))

            if output_audio is None or len(output_audio) == 0:
                raise ValueError("El audio generado está vacío.")

            output_file = "generated_music.wav"
            write(output_file, 22050, output_audio)

            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                raise ValueError("El archivo generado no se guardó correctamente.")

            self.generated_music_path = output_file
            self.play_generated_button.config(state=tk.NORMAL)
            messagebox.showinfo("Composición Generada", f"Composición guardada en {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar la composición: {e}")

    def play_generated_music(self):
        if hasattr(self, "generated_music_path") and os.path.exists(self.generated_music_path):
            try:
                # Abre el archivo con el reproductor predeterminado del sistema
                subprocess.run(['start', self.generated_music_path], shell=True, check=True)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo reproducir el archivo: {e}")
        else:
            messagebox.showerror("Error", "Primero genera un archivo de audio válido.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerApp(root)
    root.mainloop()