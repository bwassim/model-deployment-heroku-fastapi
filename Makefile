.DEFAULT_GOAL := help

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo ""
	@echo " setup 		  create a conda environment with environment.yml"
	@echo " install        install dependecies for CI/CD deployment"
	@echo " test 		  run all the tests with pytest"
	@echo " format         format all files in the starter folder"
	@echo " lint 		  lint code"
	@echo " dvc 			  run dvc pull"
	@echo " app 			  run app on local server"
	@echo " all			  run format lint and test stages"
	@echo " requests		  run the post request api"
	@echo " export_pkg     run the conda command to export only main packages without dependencies"
	@echo " Check the Makefile to know what each target is doing"

setup:
	conda env create --file environment.yml

install:
	pip install --upgrade pip && pip install -r requirements.txt

test:
	pytest -vv

format:
	black starter/*.py

lint:
	flake8 --ignore=E303,E302,E266,W503 --max-line-length=127 starter/*.py

dvc:
	dvc pull

app:
	uvicorn starter.main:app --reload --workers 1 --host 127.0.0.1 --port 8000


all:
	format lint test 

requests:
	python starter/heroku_api.py

export_pkg:
	conda env export --from-history --name deploy_project > envname.yml
