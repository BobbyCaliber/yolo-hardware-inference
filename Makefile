PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip
DOCKER      ?= docker

SRC_DIR     ?= $(CURDIR)/src
TMP_DIR     ?= $(CURDIR)/tmp
DATA_DIR    ?= $(TMP_DIR)/data
DATA_NEW    ?= $(CURDIR)/data_new
IMAGES_DIR  ?= $(SRC_DIR)/check_yolo_predict/images
WEIGHTS_DIR ?= $(TMP_DIR)/weights
MERGE_TAG   ?=
LOG_LEVEL   ?= INFO
SKIP_BUILD  ?= 0
ONLY        ?= cpu,gpu,yolo

RUN_FLAGS   := --data-dir $(DATA_DIR) --images-dir $(IMAGES_DIR) \
               --weights-dir $(WEIGHTS_DIR) \
               --log-level $(LOG_LEVEL) --only $(ONLY)
ifneq ($(SKIP_BUILD),0)
RUN_FLAGS   += --skip-build
endif

.DEFAULT_GOAL := help

.PHONY: help install build run merge run-and-merge train predict clean

help: ## show this help
	@awk 'BEGIN {FS = ":.*## "; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Variables (override as VAR=value):"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  DATA_NEW=$(DATA_NEW)"
	@echo "  IMAGES_DIR=$(IMAGES_DIR)"
	@echo "  WEIGHTS_DIR=$(WEIGHTS_DIR)"
	@echo "  ONLY=$(ONLY)            (cpu,gpu,yolo subset)"
	@echo "  SKIP_BUILD=$(SKIP_BUILD) (1 to reuse existing images)"
	@echo "  MERGE_TAG=$(MERGE_TAG)  (suffix for merged CSV filename)"

install: ## install python deps (docker SDK + training libs)
	$(PIP) install -r requirements.txt

build: ## build the three benchmark docker images
	$(DOCKER) build -t yolo-benchmark/check_cpu_config:latest    $(SRC_DIR)/check_cpu_config
	$(DOCKER) build -t yolo-benchmark/check_gpu_config:latest    $(SRC_DIR)/check_gpu_config
	$(DOCKER) build -t yolo-benchmark/check_yolo_predict:latest  $(SRC_DIR)/check_yolo_predict

run: ## run the benchmark suite (subset via ONLY=cpu,gpu,yolo)
	@mkdir -p $(DATA_DIR)
	$(PYTHON) $(SRC_DIR)/run_benchmark.py $(RUN_FLAGS)

merge: ## merge the 3 CSVs in DATA_DIR into a 35-column file in DATA_NEW
	$(PYTHON) $(SRC_DIR)/merge_results.py --data-dir $(DATA_DIR) --out-dir $(DATA_NEW) \
	    $(if $(MERGE_TAG),--tag $(MERGE_TAG))

run-and-merge: run merge ## full suite + merge into DATA_NEW

train: ## train CatBoost on data_base + data_new/*.csv, save to data_new/reg_weights_new/
	$(PYTHON) $(SRC_DIR)/train_model.py

predict: ## predict: CPU=... GPU=... RAM=... MODEL=... IMG=... BATCH=... [USED_GPU=1]
	$(PYTHON) $(SRC_DIR)/predict.py --cpu "$(CPU)" --gpu "$(GPU)" --ram $(RAM) \
	    --model $(MODEL) --img-size $(IMG) --batch $(BATCH) \
	    $(if $(filter-out 0,$(USED_GPU)),--used-gpu)

clean: ## remove tmp/, data_new/ and benchmark docker images
	rm -rf $(TMP_DIR) $(DATA_NEW)
	-$(DOCKER) image rm yolo-benchmark/check_cpu_config:latest
	-$(DOCKER) image rm yolo-benchmark/check_gpu_config:latest
	-$(DOCKER) image rm yolo-benchmark/check_yolo_predict:latest
