# Steganography Tool (Encoder & Decoder)

A Python-based tool designed to hide and retrieve encrypted payloads (text, audio, or images) within carrier image files using **steganography**. It ensures data protection using custom secret keys and optimization profiles.

## 🚀 Features
* **Multi-format Support:** Hide text (`.txt`), audio (`.wav`), and image (`.png`, `.jpg`) files.
* **Security:** Password-protected encoding and decoding using custom keys.
* **Flexible Profiles:** Performance and quality control via command-line arguments (`--profile balanced`).

---

## 🛠️ Prerequisites

Before running the scripts, ensure you have Python installed along with the required dependencies. You can install them by running:

```bash
pip install -r requirements.txt

```

💻 Usage Examples
Below are the basic commands to encode (hide) and decode (extract) different types of payloads.

📝 Hiding and Retrieving Text
To hide a text message inside a high-resolution carrier image:

```bash
# Encode text
python encoder.py --cover portadora_4k.png --payload mensaje.txt --key MI_CLAVE --out estego_texto.png --profile balanced

# Decode text
python decoder.py --stego estego_texto.png --key MI_CLAVE --out recuperado/

```

🎵 Hiding and Retrieving Audio
To hide an audio file (like a .wav) inside the carrier image:

```bash
# Encode audio
python encoder.py --cover portadora_4k.png --payload metalpipe.wav --key MI_CLAVE --out estego_audio.png --profile balanced

# Decode audio
python decoder.py --stego estego_audio.png --key MI_CLAVE --out recuperado/
```

🖼️ Hiding and Retrieving Images
To hide a secret image inside another carrier image:

```bash
# Encode image
python encoder.py --cover portadora_4k.png --payload secreto.png --key MI_CLAVE --out estego_imagen.png --profile balanced

# Decode image
python decoder.py --stego estego_imagen.png --key MI_CLAVE --out recuperado/
```
---

## 📁 Project Structure

* `encoder.py`: Main script to pack and hide the payload.
* `decoder.py`: Main script to extract the hidden content.
* `stego_core.py`: Core module handling the mathematical and processing logic for steganography.
* `stress_test.py`: Script for load testing and algorithm performance analysis.

---

## 👥 Contributors

This project was developed as a collaborative academic effort by:
* **Alejandro Muley** 
* **Carlos García** 
* **Gorka Sagristà** 
* **Aiman El Yahyaoui** 