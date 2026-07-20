from config import load_config, save_config
from setup_dialog import run_setup


def main():
    cfg = run_setup(load_config())
    save_config(cfg)


if __name__ == "__main__":
    main()
