"""Execute the tutorial's mandatory blocks against an installed distribution."""
import argparse
import json
import os
from pathlib import Path
import re
import tempfile
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import cancerxpress as cx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--database")
    parser.add_argument("--check-adaptation", action="store_true")
    args = parser.parse_args()
    if "site-packages" not in str(Path(cx.__file__).resolve()):
        raise RuntimeError("Run outside the checkout with the installed package")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["CANCERXPRESS_HOME"] = str(Path(args.resources).resolve())
    os.environ["CANCERXPRESS_TUTORIAL_OUTPUT"] = str(output / "tutorial_outputs")
    blocks = re.findall(
        r"<!-- runnable -->\s*```python\n(.*?)\n```",
        Path(args.tutorial).read_text(encoding="utf-8"), re.S,
    )
    if len(blocks) != 6:
        raise RuntimeError("Expected six mandatory tutorial blocks")
    report = {"version": version("cancerxpress"), "package": cx.__file__,
              "synthetic_tutorial_inputs": True, "blocks": []}
    namespace = {"__name__": "__tutorial__"}
    for number, code in enumerate(blocks, 1):
        start = time.monotonic()
        exec(compile(code, "%s:block%d" % (args.tutorial, number), "exec"), namespace)
        report["blocks"].append({"number": number, "status": "passed",
                                 "seconds": round(time.monotonic() - start, 2)})
    report["attribution_qc"] = namespace["attribution"].diagnostics.to_dict("records")
    model = namespace["model"]
    expression = namespace["expr"].iloc[[0]]
    clinical = namespace["clinical"].iloc[[0]]
    for name, result in (
        ("me_ig", model.attribute_me(
            expression, "MEred", steps=16, gene_id_type="ensembl")),
        ("risk_ig", model.attribute_survival_risk(
            expression, clinical, steps=16, gene_id_type="ensembl")),
    ):
        assert np.isfinite(result.attributions.to_numpy()).all()
        report[name] = {"status": "passed", "scope": "finite-output smoke check",
                        "shape": list(result.attributions.shape)}
    if args.check_adaptation:
        rng = np.random.RandomState(124)
        genes = namespace["expr"].columns
        ref_ids = ["synthetic_reference_%d" % i for i in range(12)]
        query_ids = ["synthetic_target_%d" % i for i in range(12)]
        reference_tpm = pd.DataFrame(
            rng.gamma(2, 5, size=(12, len(genes))), index=ref_ids, columns=genes)
        target_tpm = pd.DataFrame(
            rng.gamma(2, 5.2, size=(12, len(genes))), index=query_ids, columns=genes)
        ref_meta = pd.DataFrame({"cancer_type": ["BRCA"] * 12,
                                 "patient_id": ref_ids}, index=ref_ids)
        target_meta = pd.DataFrame({"cancer_type": ["BRCA"] * 12,
                                    "patient_id": query_ids}, index=query_ids)
        reference = model.build_reference(reference_tpm, ref_meta, gene_id_type="ensembl")
        fit = model.adapt(target_tpm, target_meta, reference, epochs=2, patience=2,
                          seed=42, gene_id_type="ensembl")
        with tempfile.TemporaryDirectory(dir=str(output)) as tmp:
            destination = Path(tmp) / "adapter"
            fit.save(destination)
            loaded = cx.LatentAdapter.load_adapter(destination)
            before_reload = model.encode_features(
                target_tpm, adapter=fit.adapter, metadata=target_meta,
                gene_id_type="ensembl")
            after_reload = model.encode_features(
                target_tpm, adapter=loaded, metadata=target_meta,
                gene_id_type="ensembl")
            for before, after in zip(before_reload, after_reload):
                assert np.isfinite(after.to_numpy()).all()
                np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
            hits = model.query(
                target_tpm.iloc[:1], reference, metadata=target_meta.iloc[:1],
                adapter=loaded, k=2, same_cancer=True, gene_id_type="ensembl")
            assert len(hits) == 2
        report["public_adaptation_api"] = {
            "status": "passed", "synthetic_inputs": True, "epochs": len(fit.history),
            "serialization_equivalence": True,
            "scope": "API workflow only, not biological alignment validation",
        }
    if args.reference:
        atlas = cx.ReferenceAtlas.load(args.reference)
        hits = model.query(
            namespace["expr"].iloc[[0]], atlas, space="latent", k=3,
            gene_id_type="ensembl",
        )
        assert len(hits) == 3
        assert np.isfinite(hits["distance"]).all()
        # Synthetic input is used only to check model/signature compatibility.
        report["local_atlas"] = {"samples": len(atlas.metadata), "hits": len(hits),
                                 "status": "passed"}
    if args.database:
        database = cx.SampleDatabase.from_directory(args.database)
        ids = database.sample_ids[:2]
        components = database.fetch(ids, components=[
            "raw_tpm", "corrected_expression", "latent", "me", "metadata",
        ])
        for name, frame in components.items():
            assert list(frame.index) == list(ids)
            if name != "metadata":
                assert np.isfinite(frame.to_numpy()).all()
        report["indexed_database"] = {
            "samples": len(database.sample_ids), "status": "passed",
            "component_shapes": {key: list(value.shape)
                                 for key, value in components.items()},
        }
    (output / "tutorial_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
