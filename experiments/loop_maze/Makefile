.PHONY: clean

EPOCHS ?= 5
WANDB ?= 0

loop_maze:
	python3 run.py --online-logger ${WANDB} -e ${EPOCHS}

clean:
	rm -rf nnets params scores wandb
	rm -rf *.pdf