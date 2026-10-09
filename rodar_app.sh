#!/usr/bin/env bash
# Sobe a interface web: ./rodar_app.sh  (abre http://localhost:8501)
cd "$(dirname "$0")"
( sleep 3 && open http://localhost:8501 ) &
exec .venv/bin/streamlit run app/streamlit_app.py --server.port 8501
