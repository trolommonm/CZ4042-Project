import os
import pandas as pd


def adience_preparation(data_path="data"):
    fold_indices = []
    df = pd.DataFrame(columns=["gender", "path"])
    for i in range(5):
        tmp_df = pd.read_csv(os.path.join(data_path, f"Adience/fold_{i}_data.txt"), delimiter="\t")
        # create a new column for the path to the image
        tmp_df["path"] = tmp_df["user_id"] + "/landmark_aligned_face." + \
                         tmp_df["face_id"].astype("str") + "." + tmp_df['original_image']
        # remove rows with gender == "u" and gender == NaN, keep only the gender and path columns
        tmp_df = tmp_df[["gender", "path"]][(tmp_df["gender"] != "u") & ~tmp_df["gender"].isna()]
        # convert gender into binary labels, i.e. 0 for female and 1 for male
        tmp_df["gender"] = tmp_df["gender"].apply(lambda x: {"f": "0", "m": "1"}[x])

        # concat df and tmp_df
        df = pd.concat([df, tmp_df], ignore_index=True)

        if not fold_indices:
            fold_indices.append((0, len(df) - 1))
        else:
            fold_indices.append((fold_indices[-1][1] + 1, len(df) - 1))

    for i in range(5):
        val_df = df.loc[fold_indices[i][0]:fold_indices[i][1] + 1, :]
        val_image_paths = val_df["path"].tolist()
        val_labels = val_df["gender"].tolist()

        train_df = df.loc[~df.index.isin(val_df.index), :]
        train_image_paths = train_df["path"].tolist()
        train_labels = train_df["gender"].tolist()

        fold_path = os.path.join(data_path, f"Adience/fold_{i}")
        os.makedirs(fold_path, exist_ok=True)

        with open(os.path.join(fold_path, "train.txt"), "w") as f:
            f.write("\n".join([" ".join(i) for i in zip(train_image_paths, train_labels)]))

        with open(os.path.join(fold_path, "val.txt"), "w") as f:
            f.write("\n".join([" ".join(i) for i in zip(val_image_paths, val_labels)]))


def celeba_preparation(data_path="data"):
    with open(os.path.join(data_path, "CelebA/list_attr_celeba.txt")) as f:
        lines = f.readlines()
        columns = ["filename"] + lines[1].split()
        df = pd.DataFrame([i.split() for i in lines[2:]], columns=columns)
        # drop all the other columns and keep only filename and Male
        df = df[["filename", "Male"]]
        # -1 represents female and 1 represents male in the original data, we map it to 0 for female and 1 for male
        df["Male"] = df["Male"].apply(lambda x: {"-1": "0", "1": "1"}[x])

    with open(os.path.join(data_path, "CelebA/list_eval_partition.txt")) as f:
        lines = f.readlines()
        columns = ["filename", "partition"]
        partition_df = pd.DataFrame([i.split() for i in lines[2:]], columns=columns)

    df["partition"] = partition_df["partition"]
    train_df = df[(df["partition"] == "0") | (df["partition"] == "1")]
    val_df = df[df["partition"] == "2"]

    with open(os.path.join("data/CelebA", "train.txt"), "w") as f:
        f.write("\n".join([" ".join(i) for i in zip(train_df["filename"], train_df["Male"])]))

    with open(os.path.join("data/CelebA", "test.txt"), "w") as f:
        f.write("\n".join([" ".join(i) for i in zip(val_df["filename"], val_df["Male"])]))


if __name__ == "__main__":
    adience_preparation()
    celeba_preparation()
