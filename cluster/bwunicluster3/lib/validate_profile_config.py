"""Offline guard against serving one model while recording another in CV-0.

Uses only MIRAGE's configuration layer; never constructs a backend or contacts a
server. Invoked before submission and again inside the allocation.
"""

from __future__ import annotations

import argparse

from mirage_persist.config.loader import load_config
from mirage_persist.config.schema import BackendType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("model_id")
    parser.add_argument("revision")
    parser.add_argument("served_model_name")
    parser.add_argument("family")
    args = parser.parse_args()
    config = load_config(args.config)
    if len(config.models) != 1:
        parser.error("a cluster profile must have exactly one model in its CV-0 config")
    model = config.models[0]
    # The batch job owns this local server and exports these exact variables.
    # An explicit URL would bypass that server while preserving its provenance.
    if (model.base_url is not None or model.base_url_env != "OPENAI_COMPATIBLE_BASE_URL"
            or model.api_key_env != "OPENAI_COMPATIBLE_API_KEY"):
        parser.error("cluster config must use the allocation's OPENAI_COMPATIBLE_BASE_URL/API_KEY environment")
    expected = {
        "backend": BackendType.OPENAI_COMPATIBLE,
        "model_id": args.model_id,
        "revision": args.revision,
        "served_model_name": args.served_model_name,
        "family": args.family,
    }
    for field, value in expected.items():
        actual = getattr(model, field)
        if actual != value:
            parser.error(f"profile/config mismatch: {field}: config={actual!r}, profile={value!r}")
    print(f"Config validated: {config.experiment_id} ({model.model_id}@{model.revision})")


if __name__ == "__main__":
    main()
