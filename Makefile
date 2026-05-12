PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip
DOCKER      ?= docker

SRC_DIR     ?= $(CURDIR)/src
TMP_DIR     ?= $(CURDIR)/tmp
DATA_DIR    ?= $(TMP_DIR)/data
DATA_NEW    ?= $(CURDIR)/data_new
IMAGES_DIR  ?= $(SRC_DIR)/check_model_predict/images
WEIGHTS_DIR ?= $(TMP_DIR)/weights
MERGE_TAG   ?=
LOG_LEVEL   ?= INFO
SKIP_BUILD  ?= 0
ONLY        ?= cpu,gpu,model
FAMILIES    ?= yolov5,yolov6,yolov8,yolov9,yolov10,yolo11,rtdetr,\
               detr,segformer,\
               fasterrcnn,maskrcnn,keypointrcnn,\
               deeplabv3,fcn,lraspp,\
               vit,deit,swin,efficientnet,resnet,convnext

RUN_FLAGS   := --data-dir $(DATA_DIR) --images-dir $(IMAGES_DIR) \
               --weights-dir $(WEIGHTS_DIR) \
               --log-level $(LOG_LEVEL) --only $(ONLY)
ifneq ($(SKIP_BUILD),0)
RUN_FLAGS   += --skip-build
endif

.DEFAULT_GOAL := help

.PHONY: help install build run merge train predict clean ui \
        bench-family arch-features arch-features-docker enrich collect recommend

help: ## show this help
	@awk 'BEGIN {FS = ":.*## "; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Variables (override as VAR=value):"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  DATA_NEW=$(DATA_NEW)"
	@echo "  IMAGES_DIR=$(IMAGES_DIR)"
	@echo "  WEIGHTS_DIR=$(WEIGHTS_DIR)"
	@echo "  ONLY=$(ONLY)            (cpu,gpu,model subset)"
	@echo "  FAMILIES=$(FAMILIES)"
	@echo "  SKIP_BUILD=$(SKIP_BUILD) (1 to reuse existing images)"
	@echo "  MERGE_TAG=$(MERGE_TAG)  (suffix for merged CSV filename)"

install: ## install python deps
	$(PIP) install -r requirements.txt

build: ## build the three benchmark docker images
	$(DOCKER) build -t yolo-benchmark/check_cpu_config:latest    $(SRC_DIR)/check_cpu_config
	$(DOCKER) build -t yolo-benchmark/check_gpu_config:latest    $(SRC_DIR)/check_gpu_config
	$(DOCKER) build -t yolo-benchmark/check_model_predict:latest \
	    -f $(SRC_DIR)/check_model_predict/Dockerfile $(SRC_DIR)

run: ## run the benchmark suite (cpu+gpu+model). Override ONLY=... or FAMILY=... to scope.
	@mkdir -p $(DATA_DIR)
	$(PYTHON) $(SRC_DIR)/run_benchmark.py $(RUN_FLAGS) \
	    $(if $(FAMILY),--env RUNNER_FAMILY=$(FAMILY) --env OUT_NAME=family_$(FAMILY)_predict.csv)

merge: ## merge cpu/gpu/family CSVs in DATA_DIR → one row-per-measurement file in DATA_NEW
	$(PYTHON) $(SRC_DIR)/merge_results.py --data-dir $(DATA_DIR) --out-dir $(DATA_NEW) \
	    $(if $(MERGE_TAG),--tag $(MERGE_TAG))

train: ## train CatBoost on data_base + data_new/*.csv → data_new/reg_weights_new/
	$(PYTHON) $(SRC_DIR)/train_model.py

predict: ## predict: CPU=... GPU=... [RAM=32] MODEL=... IMG=... BATCH=... [USED_GPU=1]
	$(PYTHON) $(SRC_DIR)/predict.py --cpu "$(CPU)" --gpu "$(GPU)" --ram $(or $(RAM),32) \
	    --model $(MODEL) --img-size $(IMG) --batch $(BATCH) \
	    $(if $(filter-out 0,$(USED_GPU)),--used-gpu)

ui: ## launch Streamlit frontend
	$(PYTHON) -m streamlit run $(SRC_DIR)/streamlit_app.py

recommend: ## inverse: MODEL=... IMG=... BATCH=... LATENCY=... BUDGET=... [TOP_K=10]
	$(PYTHON) $(SRC_DIR)/recommend.py --model $(MODEL) --img-size $(IMG) \
	    --batch $(BATCH) --max-latency $(LATENCY) --max-budget $(BUDGET) \
	    $(if $(TOP_K),--top-k $(TOP_K))

# ----- multi-architecture pipeline ----- #

bench-family: ## bench one family: FAMILY=rtdetr [MODELS=...] [IMG=...] [BATCH=...]
	@if [ -z "$(FAMILY)" ] && [ -z "$(MODELS)" ]; then \
	    echo "usage: make bench-family FAMILY=<name>  OR  MODELS=name1,name2  [IMG=640,800] [BATCH=1,8]"; exit 2; \
	fi
	@mkdir -p $(DATA_DIR)
	$(PYTHON) $(SRC_DIR)/run_benchmark.py --data-dir $(DATA_DIR) --images-dir $(IMAGES_DIR) \
	    --weights-dir $(WEIGHTS_DIR) --log-level $(LOG_LEVEL) --only cpu,gpu,model \
	    $(if $(filter-out 0,$(SKIP_BUILD)),--skip-build) \
	    $(if $(FAMILY),--env RUNNER_FAMILY=$(FAMILY)) \
	    $(if $(MODELS),--env MODELS=$(MODELS)) \
	    $(if $(IMG),--env IMG_SIZES=$(IMG)) \
	    $(if $(BATCH),--env BATCHES=$(BATCH)) \
	    --env OUT_NAME=family_$(or $(FAMILY),custom)_predict.csv

arch-features: ## profile arch features for all registered models (or FAMILY=...)
	$(PYTHON) scripts/compute_arch_features.py --append \
	    $(if $(FAMILY),--family $(FAMILY)) \
	    $(if $(MODELS),--models $(MODELS))

arch-features-docker: ## same, but runs inside check_model_predict container (has timm/transformers/etc.)
	$(DOCKER) run --rm --gpus all \
	    -u $$(id -u):$$(id -g) \
	    -e HOME=/tmp -e USER=appuser -e LOGNAME=appuser \
	    -e HF_HOME=/tmp/.hf -e XDG_CACHE_HOME=/tmp/.cache \
	    -v $(CURDIR):/app/host \
	    -w /app/host \
	    --entrypoint $(PYTHON) \
	    yolo-benchmark/check_model_predict:latest \
	    scripts/compute_arch_features.py --append \
	    $(if $(FAMILY),--family $(FAMILY)) \
	    $(if $(MODELS),--models $(MODELS))

enrich: ## enrich data_base.csv with arch features + roofline
	$(PYTHON) scripts/enrich_dataset.py

# ----- one-shot reproduction on a single platform ----- #

collect: build ## ONE-SHOT: bench every family in FAMILIES → merge → train (host descriptors collected once)
	@mkdir -p $(DATA_DIR)
	@date +%s > $(DATA_DIR)/.collect_start
	@first=1; for fam in $(subst $(comma), ,$(FAMILIES)); do \
	    if [ $$first -eq 1 ]; then \
	        echo "[collect] family=$$fam (cpu+gpu+model)"; \
	        $(PYTHON) $(SRC_DIR)/run_benchmark.py --data-dir $(DATA_DIR) \
	            --images-dir $(IMAGES_DIR) --weights-dir $(WEIGHTS_DIR) \
	            --log-level $(LOG_LEVEL) --only cpu,gpu,model --skip-build \
	            --env RUNNER_FAMILY=$$fam --env OUT_NAME=family_$${fam}_predict.csv || exit 1; \
	        first=0; \
	    else \
	        echo "[collect] family=$$fam (model only)"; \
	        $(PYTHON) $(SRC_DIR)/run_benchmark.py --data-dir $(DATA_DIR) \
	            --images-dir $(IMAGES_DIR) --weights-dir $(WEIGHTS_DIR) \
	            --log-level $(LOG_LEVEL) --only model --skip-build \
	            --env RUNNER_FAMILY=$$fam --env OUT_NAME=family_$${fam}_predict.csv || exit 1; \
	    fi; \
	done
	$(PYTHON) $(SRC_DIR)/merge_results.py --data-dir $(DATA_DIR) --out-dir $(DATA_NEW) \
	    --tag $$(hostname)
	@$(MAKE) --no-print-directory arch-features-docker
	$(PYTHON) $(SRC_DIR)/train_model.py
	@end=$$(date +%s); start=$$(cat $(DATA_DIR)/.collect_start); elapsed=$$((end-start)); \
	    h=$$((elapsed/3600)); m=$$(((elapsed%3600)/60)); s=$$((elapsed%60)); \
	    printf "[collect] total elapsed: %02d:%02d:%02d (%ds)\n" $$h $$m $$s $$elapsed
	@rm -f $(DATA_DIR)/.collect_start
	@echo "[collect] done — fresh weights in $(DATA_NEW)/reg_weights_new/"
	@echo "[collect] tip: subset families via FAMILIES=rtdetr (default benches all 6 YOLO generations + RT-DETR)"

comma := ,

clean: ## remove tmp/, data_new/ and benchmark docker images
	rm -rf $(TMP_DIR) $(DATA_NEW)
	-$(DOCKER) image rm yolo-benchmark/check_cpu_config:latest
	-$(DOCKER) image rm yolo-benchmark/check_gpu_config:latest
	-$(DOCKER) image rm yolo-benchmark/check_model_predict:latest
