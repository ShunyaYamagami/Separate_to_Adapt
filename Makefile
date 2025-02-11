ENV_NAME := sta
# PYTHON_VERSION := 3.6
PYTHON_VERSION := 3.10

# create_envターゲット: condaで新しい環境を作成して依存関係をインストール
create_env:
	@echo "Updating conda..."
	conda update -n base -c defaults conda -y
	@echo "Creating environment..."
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y
	@echo "Installing python dependencies..."
	conda run -n $(ENV_NAME) conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
	conda run -n $(ENV_NAME) pip install -r requirements.txt
	cp -r /nas/data/syamagami/GDA/data/GDA_DA_methods/data ./
	cp -r /nas/data/syamagami/GDA/data/GDA_DA_methods/Separate_to_Adapt/pretrained_models ./

# remove_envターゲット: condaの環境を削除
remove_env:
	@echo "Removing environment..."
	conda env remove --name $(ENV_NAME)
	@echo "Environment removed successfully."