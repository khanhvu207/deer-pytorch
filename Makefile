.PHONY: clean

main:
	python3 main.py --seed=2022
#	python3 main.py --seed=2023
#	python3 main.py --seed=2024
#	python3 main.py --seed=2025
#	python3 main.py --seed=2026

clean:
	rm -rf nnets params scores wandb
	rm -rf *.pdf
	rm -rf *.pkl