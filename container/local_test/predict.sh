#!/bin/bash

payload=$1
content=${2:-application/json}

curl -H "Content-Type: ${content}" -d "@${payload}" -X POST http://localhost:8080/invocations