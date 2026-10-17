FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# The base image ships a Yarn apt repo with a missing signing key; that breaks
# `apt-get update` and with it the docker-in-docker feature install.
RUN rm -f /etc/apt/sources.list.d/yarn*
