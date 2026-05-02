"""Lightweight JSON cache for repeated questions and answers."""

import json
import os

CACHE_FILE = "cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_cached(query):
    cache = load_cache()
    return cache.get(query.strip().lower())

def set_cache(query, answer, sources):
    cache = load_cache()
    cache[query.strip().lower()] = {
        "answer": answer,
        "sources": sources
    }
    save_cache(cache)