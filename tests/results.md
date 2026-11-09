# Example results

Generated on 2026-11-09 with `nomic-embed-text` and `qwen3:4b-instruct`, top 3 chunks.
The chat model is not deterministic, so answers can change slightly between runs.
Regenerate with `uv run python -m tests.run_examples`.

## What is the QBER threshold for BB84?

- **Expected:** QBER
- **Retrieved:** QBER (0.754), BB84 (0.733), Monitoring & anomaly detection (0.601)

**Answer:**

> The QBER threshold for BB84 is approximately 11%.

## How much light is lost per kilometre in optical fibre?

- **Expected:** Secret key rate vs distance
- **Retrieved:** Secret key rate vs distance (0.678), Quantum repeaters & satellites (0.528), Decoy states (0.506)

**Answer:**

> The context states that in standard telecom optical fibre at wavelengths near 1550 nm, loss is approximately 0.2 dB/km. However, it does not provide a direct conversion of this loss (in decibels per kilometre) into a percentage or amount of light lost per kilometre in terms of actual light intensity. While it mentions that "only about 1% of the light survives every 100 km," this is a consequence of the exponential loss, not a direct figure for per kilometre loss.
>
> Therefore, based on the context, we know the loss is approximately 0.2 dB/km, but we cannot directly convert this to a precise percentage or amount of light lost per kilometre without additional assumptions or calculations not provided in the text.
>
> Answer: The context does not provide a direct figure for how much light is lost per kilometre in optical fibre in terms of percentage or actual intensity. It only states the loss is approximately 0.2 dB/km, which is a logarithmic measure and not a direct percentage. Thus, based on the context, we cannot give a precise amount of light lost per kilometre.

## Which machine learning models can detect anomalies in QKD telemetry?

- **Expected:** Monitoring & anomaly detection
- **Retrieved:** Monitoring & anomaly detection (0.796), BB84 (0.662), What is QKD (0.660)

**Answer:**

> The machine learning models that can detect anomalies in QKD telemetry are Isolation Forest and Autoencoder.

## What QBER does an intercept-resend attack on every signal cause?

- **Expected:** Intercept-resend attack
- **Retrieved:** Intercept-resend attack (0.813), Monitoring & anomaly detection (0.676), QBER (0.639)

**Answer:**

> An intercept-resend attack on every signal causes a QBER increase of about 25 percentage points.

## What is the capital of France?

- **Expected:** nothing relevant
- **Retrieved:** Quantum repeaters & satellites (0.417), E91 (0.380), What is QKD (0.373)

**Answer:**

> I don't know.
