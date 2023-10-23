# Imgenie-airflow

## Setup

### Dependency install

```base
poetry install
```

### Huggingface setup

```bash
apt install git-lfs
git lfs install
mkdir -p data/{crawler,finetune,preprocess}
git clone https://<username>:<token>@huggingface.co/datasets/RecDol/PLAYLIST_airflow data/dataset
```
