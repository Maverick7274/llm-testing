# Large Language Model Testing

This repository contains the code for testing the Large Language Model (LLM) developed by me. The LLM is a transformer-based model that is trained on a large corpus of text data. The model is capable of generating human-like text and can be fine-tuned for specific tasks.

## How to run the LLM

* To run the LLM, you need to have the following dependencies installed:

```plaintext
matplotlib
numpy
pylzma
ipykernel
jupyter
torch==2.0.0+cu118 -f https://download.pytorch.org/whl/cu118
```

* Before installing the dependencies, you need to create a virtual environment. You can create a virtual environment using the following command:

```bash
python -m venv venv
```

* After creating the virtual environment, you can activate it using the following command:

```bash
source venv/bin/activate
```

> **Note:** The above command is for Linux and MacOS.

* If you are using Windows, you can activate the virtual environment using the following command:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUse
.\venv\Scripts\Activate.ps1
```

* You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

* After installing the dependencies, you will need to install the venv as a Jupyter kernel. You can do this using the following command:

```bash
python -m ipykernel install --user --name=venv --display-name "your-display-name"
```
