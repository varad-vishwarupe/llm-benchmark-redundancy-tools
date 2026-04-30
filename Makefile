# Run from repo root: `make all`
# All scripts live in code/ and reference ../data/ and ../figures/

.PHONY: all clean install hf livebench figures

all: hf livebench figures

hf: hf-pca hf-imputation hf-gbm hf-robustness hf-twostage hf-floor hf-power

hf-pca:
	cd code && python3 hf_pca.py

hf-imputation:
	cd code && python3 hf_imputation.py

hf-gbm:
	cd code && python3 hf_gbm.py

hf-robustness:
	cd code && python3 hf_robustness.py

hf-twostage:
	cd code && python3 hf_two_stage.py

hf-floor:
	cd code && python3 hf_floor_effects.py

hf-power:
	cd code && python3 hf_power_analysis.py

livebench: livebench-analysis livebench-bootstrap

livebench-analysis:
	cd code && python3 livebench_analysis.py

livebench-bootstrap:
	cd code && python3 livebench_bootstrap.py

figures:
	cd code && python3 make_figures.py

clean:
	rm -f data/hf_imputation_full.csv data/hf_two_stage_pareto.csv data/hf_power_analysis.csv

install:
	pip install -r code/requirements.txt
