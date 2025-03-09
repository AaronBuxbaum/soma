## Soma Cell Segmentation

How to use:
1. Download this repository (click Code -> Download ZIP)
2. Unzip the downloaded file
3. Open `Terminal` and navigate to the unzipped folder (e.g. `cd ~/Desktop/soma`)
4. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
5. Install dependencies: `poetry install`
6. Copy your `.tif` files to the `input` folder
7. Run command: `poetry run python soma.py`

Note that you can supply all kinds of options to the command to customize how it works. For example, you can specify the output folder, microns per pixel, and various optional preprocessing steps. Run `poetry run python soma.py --help` to see all available options. Also see [PARAMETERS.md](PARAMETERS.md) for more details.
