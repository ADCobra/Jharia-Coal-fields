PY ?= python3

.PHONY: install models demo serve test eval docker deploy clean

install:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install --no-deps -e .

models:
	firewatch download-models

demo:
	firewatch demo --fresh

serve:
	firewatch serve

test:
	pytest

eval:
	firewatch eval all --out reports

docker:
	docker build -t jharia-fire-watch:local .

deploy:
	infra/deploy.sh

clean:
	firewatch reset
