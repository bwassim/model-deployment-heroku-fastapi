install:
		pip install --upgrade pip && pip install -r requirements.txt

test:
		pytest -vv

format:
		black starter/*.py

lint:
		flake8 --ignore=E303,E302  --max-line-length=127 starter/*.py

dvc:
		dvc pull -r s3remote