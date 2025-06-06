########################################################################
# 1) Base image: Python 3.10-slim
########################################################################
FROM python:3.10-slim

########################################################################
# 2) Disable .pyc writes and enable unbuffered logging
########################################################################
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

########################################################################
# 3) Install OS-level dependencies
########################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      build-essential \
      git \
      curl && \
    rm -rf /var/lib/apt/lists/*

########################################################################
# 4) Copy and install Python dependencies AS ROOT
########################################################################
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

########################################################################
# 5) Create a non-root user
########################################################################
ARG USER=appuser
ARG UID=1001
RUN useradd --create-home --uid $UID $USER

# ────────────────────────────────────────────────────────────────────────
# 5a) Make sure the `user_uploads` folder exists and is owned by appuser
#      before switching to appuser.
# ────────────────────────────────────────────────────────────────────────
RUN mkdir -p /home/$USER/app/project/models/user_uploads && \
    chown -R $USER:$USER /home/$USER/app/project/models/user_uploads

# ────────────────────────────────────────────────────────────────────────
# 5b) Now set WORKDIR and switch to appuser
# ────────────────────────────────────────────────────────────────────────
WORKDIR /home/$USER/app
USER $USER

########################################################################
# 6) Copy application code (as appuser) — now user_uploads is already owned
########################################################################
COPY --chown=$USER:$USER main_api.py .
COPY --chown=$USER:$USER project/ ./project
COPY --chown=$USER:$USER project/models/model_weights ./project/models/model_weights

########################################################################
# 7) Expose FastAPI port
########################################################################
EXPOSE 8000

########################################################################
# 8) Default command: launch FastAPI via uvicorn
########################################################################
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
