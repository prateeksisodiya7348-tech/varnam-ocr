"""Varnam OCR API package.

Layering (strict, one direction only):

    routes/          HTTP concerns: parsing, status codes, response models
      -> services/   business logic: pipeline, engine, storage, validation
        -> core/     cross-cutting: config, errors, logging, rate limiting

Nothing in ``services`` imports from ``routes``; that is what makes the
pipeline testable without an HTTP client or a running model.
"""

__version__ = "1.0.0"
