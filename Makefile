PYTHON ?= python3
.PHONY: setup demo chat eval test check
setup:
	$(PYTHON) run.py setup
demo:
	$(PYTHON) run.py demo
chat:
	$(PYTHON) run.py chat
eval:
	$(PYTHON) run.py eval
test:
	$(PYTHON) run.py test
check:
	$(PYTHON) run.py check
