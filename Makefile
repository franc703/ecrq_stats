install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black-nb notebooks/*.ipynb

all: install format 