#!/usr/bin/env bash

set -eux

cd tests
pytest unit_test -v
