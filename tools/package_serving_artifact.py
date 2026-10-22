"""Remove the unused Keras LSTM object from this project's trusted T6 pickle.

This is packaging, not retraining: calibrated LogReg, RF, XGBoost, class
encoder, scaler and thresholds are preserved. The input must be the supplied
project artifact, not an untrusted pickle. Its source hash is recorded.
"""
import argparse
import hashlib
import pickle
from pathlib import Path


class UnusedSequential:
    @staticmethod
    def _unpickle_model(stream):
        return None


class ServingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "keras.src.models.sequential" and name == "Sequential":
            return UnusedSequential
        return super().find_class(module, name)


def package(source, destination):
    with source.open("rb") as stream:
        artifact = ServingUnpickler(stream).load()
    removed = [key for key in artifact if key.startswith("lstm_")]
    for key in removed:
        artifact.pop(key)
    artifact["source_artifact_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact["packaging_note"] = "Unused LSTM fields excluded; calibrated serving estimators preserved."
    artifact["excluded_fields"] = removed
    with destination.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=5)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    package(args.source, args.destination)
