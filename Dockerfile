# CAD Kinematics: STEP -> joint extraction -> Godot-ready robot.json.
# conda-forge FreeCAD (not Ubuntu's apt package — verified more reliable,
# see godot_transformer's Dockerfile notes) + numpy in the same env, so both
# ui_app.py (stdlib) and pipeline.py (FreeCAD + numpy) work from one image.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" -O /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p /opt/conda \
    && rm /tmp/mf.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda create -y -n fc -c conda-forge freecad python=3.11 numpy && conda clean -afy
ENV PATH=/opt/conda/envs/fc/bin:$PATH

WORKDIR /app
COPY src/ src/
COPY pipeline/ pipeline/
COPY ui/ ui/
COPY viewer/ viewer/
COPY schemas/ schemas/
COPY fixtures/ fixtures/
COPY pipeline.py run_with_freecad.sh ui_app.py ./
RUN chmod +x run_with_freecad.sh

# Container listens on all interfaces so Docker's published ports reach it;
# host runs (./start_ui.sh) are untouched and keep defaulting to 127.0.0.1.
ENV UI_BIND=0.0.0.0

EXPOSE 8787 8765
CMD ["python", "ui_app.py"]
