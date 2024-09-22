#!/bin/bash


case $1 in
    frontend)
        cd webapp
        ng serve
        ;;
    backend)
        uvicorn alpha_cc.api.endpoints:app --host 0.0.0.0 --reload
        ;;
    redis)
        docker compose -f docker-compose.webapp.yaml up redis
        ;;
    *)
        echo "Usage: run-app.sh <frontend|backend|redis>"
        exit 1
        ;;
esac
