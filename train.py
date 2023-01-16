import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="pointcloud_models/config/default_config.yaml",
    )

    args = parser.parse_args()
    train(args)
