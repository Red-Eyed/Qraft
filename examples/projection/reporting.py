"""Save a portable report of projection errors and selected ONNX artifacts."""

from examples.projection.schema import Report


def save(report: Report) -> None:
    """Persist JSON provenance and human tables with report-relative model links."""
    directory = report.config.output
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(report.model_dump_json(indent=2))
    rows = "".join(
        f"<tr><td>{variant.method.value}</td>"
        f"<td>{variant.metrics.mean_squared_error:.6g}</td>"
        f"<td>{variant.metrics.maximum_absolute_error:.6g}</td>"
        f'<td><a href="{variant.artifact.as_posix()}">ONNX</a> · '
        f'<a href="{variant.method.value}.coverage.json">Coverage</a></td></tr>'
        for variant in report.variants
    )
    (directory / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>QuantSmith · Projection</title><style>"
        "body{max-width:900px;margin:40px auto;padding:0 20px;font:16px system-ui}"
        "table{border-collapse:collapse}td,th{padding:12px;text-align:left;"
        "border-bottom:1px solid #aaa}</style>"
        "<h1>Projection × quantization methods</h1>"
        "<p>Every recipe starts from the same deterministic FP32 MatMul. "
        "Calibration and held-out batches use separate random seeds.</p>"
        '<p><a href="onnx_fp32.onnx">FP32 ONNX</a> · '
        '<a href="results.json">Configuration and measurements</a></p>'
        "<table><tr><th>Method</th><th>Mean squared error</th>"
        f"<th>Maximum absolute error</th><th>Artifacts</th></tr>{rows}</table></html>"
    )
    lines = [
        "# Projection × quantization methods\n",
        "| Method | Mean squared error | Maximum absolute error |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {variant.method.value} | {variant.metrics.mean_squared_error:.6g} | "
        f"{variant.metrics.maximum_absolute_error:.6g} |"
        for variant in report.variants
    )
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
