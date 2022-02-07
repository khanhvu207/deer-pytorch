.PHONY: clean

main:
	python3 main.py

clean:
	rm -rf nnets params scores wandb
	rm -rf *.pdf