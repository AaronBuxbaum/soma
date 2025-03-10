## Soma Cell Segmentation

Install:
1. Open the `Terminal` app -- you can find it under Applications > Utilities
2. Install Git: `git version` (this may prompt you to install)
3. Download: `cd ~/Desktop && git clone https://github.com/AaronBuxbaum/soma.git`

Setup:
1. Open `Terminal` and run `cd ~/Desktop/soma`
2. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
  * there may be steps that it asks you to do at the end. If there are, do them.
3. Install Poetry: `brew install poetry`
4. Install dependencies: `poetry install`

Run:
1. Open `Terminal` and run `cd ~/Desktop/soma`
2. Copy your `.tif` files to the `input` folder
3. Run it!: `poetry run python soma.py`

Note that you can supply all kinds of options to the command to customize how it works. For example, you can specify the output folder, microns per pixel, and various optional preprocessing steps. Run `poetry run python soma.py --help` to see all available options. Also see [`PARAMETERS.md`](PARAMETERS.md) for more details.


---

If you're having trouble with installation, try these alternative steps:
1. Download: visit [https://github.com/AaronBuxbaum/soma/archive/refs/heads/main.zip](https://github.com/AaronBuxbaum/soma/archive/refs/heads/main.zip) in your browser
2. Open the `Terminal` app -- you can find it under Applications > Utilities
3. Set up the file: `unzip ~/Downloads/soma-main.zip && mv ~/Downloads/soma-main ~/Desktop/soma && cd ~/Desktop/soma`
