# Tracking API

## Position feed

The tracking API exposes a position endpoint at /v1/shipments/{id}/position. The
embedded GPS device sends a ping every 2 minutes while the vehicle is moving. A
parked vehicle sends a ping every 15 minutes.

## Limits

The public API allows 600 requests per minute per key. The service targets 99.5%
uptime measured monthly.

## Authentication

Requests use a bearer token. Tokens expire every 90 days and must be rotated by
the integrating customer.

## Errors

A missing shipment returns HTTP 404. A rate-limited request returns HTTP 429 with
a Retry-After header in seconds.
