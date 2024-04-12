#!/bin/bash

uvicorn alpha_cc.api.endpoints:app --host 0.0.0.0 --reload
