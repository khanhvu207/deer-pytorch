.PHONY: clean

EPOCHS ?= 50
WANDB ?= 0

labyrinth:
	python3 run_simple_maze_pytorch.py

snake:
	python3 run_simple_snake.py --online-logger ${WANDB} -e ${EPOCHS}

clean:
	rm -rf nnets params scores wandb
	rm -rf *.pdf