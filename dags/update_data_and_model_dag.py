import os
import dotenv
from os import path
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from typing import Literal

dotenv.load_dotenv()

# config
CR_SPOTIFY_CID: str = os.environ.get("CR_SPOTIFY_CID")
CR_SPOTIFY_PWD: str = os.environ.get("CR_SPOTIFY_PWD")
CR_DB_HOST: str = os.environ.get("CR_DB_HOST")
CR_DB_NAME: str = os.environ.get("CR_DB_NAME")
CR_DB_USERNAME: str = os.environ.get("CR_DB_USERNAME")
CR_DB_PASSWORD: str = os.environ.get("CR_DB_PASSWORD")
ML_WANDB_API_KEY: str = os.environ.get("ML_WANDB_API_KEY")

TagType = Literal["mood", "weather", "sit"]


def cur_datestamp():
    cur = datetime.now()
    return f"{cur.year}_{cur.month}_{cur.day}"


def get_crawl_genie_data_task(output_path: str, output_pl_file: str) -> BashOperator:
    return BashOperator(
        task_id="crawl_genie_data_task",
        bash_command="python crawler.py",
        cwd="genie_crawler",
        env={
            "output_path": output_path,
            "output_pl_file": output_pl_file,
            "enable_output_csv": "true",
            "SPOTIFY_CID": CR_SPOTIFY_CID,
            "SPOTIFY_PWD": CR_SPOTIFY_PWD,
            "DB_HOST": CR_DB_HOST,
            "DB_NAME": CR_DB_NAME,
            "DB_USERNAME": CR_DB_USERNAME,
            "DB_PASSWORD": CR_DB_PASSWORD,
        },
        append_env=True,
    )


def get_run_preprocessing_task(tag_type: TagType, train_file: str):
    return BashOperator(
        task_id=f"run_preprocess_{tag_type}_task",
        cwd="ml",
        bash_command="python run_preprocessing.py " f"train_file='{train_file}' " f"tag_type='{tag_type}' ",
    )


def get_run_finetune_task(tag_type: TagType, timestamp: str, wandb_project: str):
    return BashOperator(
        cwd="ml",
        task_id=f"run_finetune_{tag_type}_task",
        bash_command="python run_finetune.py "
        f"timestamp='{timestamp}' "
        f"wandb.project='{wandb_project}' "
        f"data.name='{tag_type}' "
        f"data.tag_type='{tag_type}' "
        f"trainer.accelerator='cpu' ",
        append_env=True,
        env={"WANDB_API_KEY": ML_WANDB_API_KEY},
    )


def get_run_indexing_task(tag_type: TagType):
    return BashOperator(
        cwd="ml",
        task_id=f"run_indexing_{tag_type}_task",
        bash_command=f"python run_indexing.py " f"tag_type='{tag_type}' " f"name='{tag_type}' ",
    )


default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "start_date": datetime(2023, 9, 1),
    "retires": 1,
    "retry_delay": timedelta(minutes=5),
}


CUR_DATESTAMP = cur_datestamp()
data_dir = path.abspath("data")
os.makedirs(data_dir, exist_ok=True)

with DAG(
    dag_id="update_data_and_model_dag",
    default_args=default_args,
    schedule="0 0 1 * *",
    tags=["data"],
):
    # crawler
    crawler_dir = path.join(data_dir, "crawler")
    crawler_pl_file = f"playlist_{CUR_DATESTAMP}.csv"
    crawler_pl_path = path.join(crawler_dir, crawler_pl_file)

    crawl_genie_data_task = get_crawl_genie_data_task(data_dir, crawler_pl_path)

    # preprocessing
    run_preprocessing_mood_task = get_run_preprocessing_task("mood", crawler_pl_file)
    run_preprocessing_weather_task = get_run_preprocessing_task("weather", crawler_pl_file)
    run_preprocessing_sit_task = get_run_preprocessing_task("sit", crawler_pl_file)

    # finetune
    wandb_project = f"AIRFLOW_{CUR_DATESTAMP}"
    run_finetune_mood_task = get_run_finetune_task("mood", CUR_DATESTAMP, wandb_project)
    run_finetune_weather_task = get_run_finetune_task("weather", CUR_DATESTAMP, wandb_project)
    run_finetune_sit_task = get_run_finetune_task("sit", CUR_DATESTAMP, wandb_project)

    # indexing
    run_indexing_mood_task = get_run_indexing_task("mood")
    run_indexing_weather_task = get_run_indexing_task("weather")
    run_indexing_sit_task = get_run_indexing_task("sit")

    crawl_genie_data_task >> [
        run_preprocessing_mood_task,
        run_preprocessing_weather_task,
        run_preprocessing_sit_task,
    ]

    run_preprocessing_mood_task >> run_finetune_mood_task >> run_indexing_mood_task
    run_preprocessing_weather_task >> run_finetune_weather_task >> run_indexing_weather_task
    run_preprocessing_sit_task >> run_finetune_sit_task >> run_indexing_sit_task
