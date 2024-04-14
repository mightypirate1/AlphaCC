#!/bin/bash


case $1 in
    frontend)
        cd webapp
        ng serve
        ;;
    backend)
        uvicorn alpha_cc.api.endpoints:app --host 0.0.0.0 --reload
        ;;
    *)
        echo "Usage: run-app.sh <frontend|backend>"
        exit 1
        ;;
esac
