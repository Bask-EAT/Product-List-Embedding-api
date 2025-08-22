SHELL := /bin/bash

COMPOSE := docker compose -f /mnt/data/dockerized/docker-compose.yml
COMPOSE_DEV := docker compose -f /mnt/data/dockerized/docker-compose.dev.yml

.PHONY: up up-dev down logs rebuild clean

up:
	$(COMPOSE) up -d --build

up-dev:
	$(COMPOSE_DEV) up -d --build

down:
	$(COMPOSE) down || true
	$(COMPOSE_DEV) down || true

logs:
	$(COMPOSE) logs -f || $(COMPOSE_DEV) logs -f

rebuild:
	$(COMPOSE) build --no-cache
	$(COMPOSE_DEV) build --no-cache

clean:
	docker network rm ple-net 2>/dev/null || true
	docker rm -f ple-backend ple-backend-dev 2>/dev/null || true
