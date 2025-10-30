#!/bin/zsh

# Avvia Docker Desktop se non è già attivo
if ! pgrep -x "Docker" > /dev/null; then
  echo "⏳ Avvio Docker Desktop..."
  open -a Docker
  # aspetta che parta
  while ! docker system info > /dev/null 2>&1; do
    sleep 2
  done
fi

# Vai nella cartella del progetto
cd "$(dirname "$0")"

# Avvia il container
echo "🚀 Avvio del container Jupyter..."
docker run -it --rm -p 8888:8888 -v "$PWD":/app daniele_dl_thesis

