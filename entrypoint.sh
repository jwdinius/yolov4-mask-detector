#!/bin/bash
set -e

# start jackd server to avoid webcam crash with guvcview
jackd -d dummy &
exec "$@"
