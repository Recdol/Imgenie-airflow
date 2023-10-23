import os
import hydra
import datasets
import pandas as pd
from src.utils import set_seed
from huggingface_hub import HfApi
from src.preprocess import preprocess_data, generate_df, check_img_temp


def main(config) -> None:
    # set seed
    set_seed(config.seed)

    # csv to pd.DataFrame (version == playlists.csv)
    data_dir = config.data_dir
    output_dir = config.output_dir
    train_file = config.train_file
    tag_path = config.tag_path
    tag_type = config.tag_type
    repo_id = config.repo_id
    name = config.dataset_name
    save_dir = os.path.join(output_dir, f"{tag_type}_dataset/{name}")
    data_path = os.path.join(data_dir, train_file)

    # generate subset dataset (tag == [weather, situation, mood])
    pd.set_option("mode.chained_assignment", None)

    df = pd.read_csv(data_path)
    tag_df = pd.read_csv(tag_path)
    df = preprocess_data(df, tag_df)
    data = generate_df(df, tag_type)

    # pd.DataFrame to datasets.Dataset
    dataset = datasets.Dataset.from_pandas(data)

    # map url to PIL.Image
    dataset = dataset.map(lambda x: {"image": check_img_temp(x["playlist_img_url"][:-22], config.temp_dir)})

    # save dataset
    dataset = dataset.remove_columns("__index_level_0__")
    dataset.save_to_disk(save_dir)

    # Upload to Huggingface Hub
    api = HfApi()
    api.upload_folder(
        folder_path=save_dir,
        path_in_repo=f"{tag_type}/{name}",
        repo_id=repo_id,
        commit_message=f"upload dataset: {name}",
        repo_type="dataset",
    )


@hydra.main(version_base="1.2", config_path="configs/preprocessing", config_name="config.yaml")
def main_hydra(config) -> None:
    main(config)


if __name__ == "__main__":
    main_hydra()
