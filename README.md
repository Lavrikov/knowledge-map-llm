# knowledge-map-llm
Knowledge map implementation with LLM

This is an implementation of knowledge map for logs analysis with LLM. 
Knowledge map is an approach of process you data with LLM.
You can get answers about any amount of your data with LLM with relatively rare occasion of gallucinations.
Method is quite opposite to common RAG solutions.
It has hand made analysis part to create knowledge map.

log_example1.py - create database with errors log from pytorch github issues and questions.
This questions represents knowledge map e.q. the way professional human investivating logs to find cause of errors.
run.py - load qwen2.5-32B-Instruct quantized to 4bit model from HF and run analysis according to questions
