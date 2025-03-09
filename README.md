## Soma Cell Segmentation

How to use:
1. Download: visit [https://github.com/AaronBuxbaum/soma/archive/refs/heads/main.zip](https://github.com/AaronBuxbaum/soma/archive/refs/heads/main.zip) in your browser
2. Open the `Terminal` app (you can find it under Applications > Utilities)
3. Set up the file: `unzip ~/Downloads/soma-main.zip && mv ~/Downloads/soma-main ~/Desktop/soma && cd ~/Desktop/soma`
4. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
5. Install dependencies: `poetry install`
6. Copy your `.tif` files to the `input` folder
7. Run it!: `poetry run python soma.py`

Note that you can supply all kinds of options to the command to customize how it works. For example, you can specify the output folder, microns per pixel, and various optional preprocessing steps. Run `poetry run python soma.py --help` to see all available options. Also see [`PARAMETERS.md`](PARAMETERS.md) for more details.
