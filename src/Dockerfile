# Usa un'immagine base ufficiale con Python
FROM python:3.12-slim

# Imposta la working directory dentro il container
WORKDIR /app

# Copia i file necessari
COPY requirements.txt .
COPY . .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta per JupyterLab
EXPOSE 8888

# Comando di default: avvia JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
