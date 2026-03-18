Below is a self-contained research document you can drop into a repo (README or /docs/research.md).
It’s structured so a developer or researcher can reconstruct the entire context and start implementing the codebase.

⸻

Synthetic Aperture Radar Doppler Tomography

Reproduction Research Document

Goal:
Recreate the SAR Doppler tomography algorithm described by Filippo Biondi in order to:
	1.	Reproduce the imaging pipeline described in the paper
	2.	Validate whether similar results can be obtained from publicly available SAR datasets
	3.	Optimize the algorithm for real-time tomographic rendering

This document provides the research context, algorithm reconstruction, dataset sources, and engineering roadmap needed to build the implementation.

⸻

1. Background

Synthetic Aperture Radar (SAR)

Synthetic Aperture Radar (SAR) is a radar imaging technique that synthesizes a large antenna by combining radar pulses collected along the satellite flight path.

Instead of using a physically large antenna, the radar platform:
	1.	Transmits pulses continuously
	2.	Moves along an orbital path
	3.	Records phase history of the reflected signals

The motion of the satellite creates an effective antenna aperture that can be hundreds of meters long.

This allows spaceborne radars to achieve meter-level resolution.

Key references:
	•	Synthetic Aperture Radar

Typical orbital parameters:

parameter	value
satellite speed	~7 km/s
synthetic aperture duration	5–20 seconds
distance traveled during aperture	35–140 km


⸻

2. Biondi Doppler Tomography

The research method described by Biondi proposes using Doppler sub-apertures extracted from a single SAR acquisition to reconstruct a tomographic signal.

Instead of relying on multiple satellite passes, the algorithm analyzes the temporal diversity inside one synthetic aperture.

Core idea

A focused SAR SLC image contains the full Doppler spectrum of the radar echoes collected during the satellite pass.

By slicing this Doppler spectrum into sub-apertures we obtain multiple lower-resolution images corresponding to different viewing geometries.

These sub-aperture images can be analyzed to detect micro-motion signals.

Examples of motion sources:
	•	structural vibration
	•	seismic microtremor
	•	elastic deformation

These signals are then inverted using a tomographic reconstruction operator.

⸻

3. Data Representation

The SAR data product required for this experiment is:

Single Look Complex (SLC)

SLC images contain:
	•	complex amplitude
	•	phase information
	•	full azimuth aperture

Typical SLC representation:

SLC[r, a] ∈ ℂ

where

r = range dimension
a = azimuth dimension

The azimuth dimension contains the synthetic aperture information.

⸻

4. Algorithm Reconstruction

Based on the published description, the pipeline can be reconstructed as follows.

Stage 1: Input SLC

SLC[r, a]

This is a focused complex SAR image.

⸻

Stage 2: Doppler Transform

Compute the 2D Fourier transform:

S = FFT2(SLC)

This exposes the Doppler frequency spectrum.

⸻

Stage 3: Sub-Aperture Decomposition

Create K Doppler windows:

W_k(f)

Each window isolates a subset of the Doppler bandwidth.

S_k = S * W_k


⸻

Stage 4: Reconstruct Sub-Aperture Images

Inverse transform:

SLC_k = IFFT2(S_k)

Each SLC_k corresponds to a smaller synthetic aperture.

Resolution decreases but viewing geometry changes.

⸻

Stage 5: Pixel Motion Estimation

For each pixel:

y_k = phase_or_shift(SLC_k)

This produces a measurement vector:

Y = [y_1, y_2, ... y_K]


⸻

Stage 6: Tomographic Model

The measurement model is:

Y = A(Kz, z) h(z)

Where:

A[k,n] = exp(i * Kz[k] * z[n])

and

Kz = 4π B⊥ / (λ r sinθ)

Variables:

symbol	meaning
λ	radar wavelength
B⊥	perpendicular baseline
r	slant range
θ	incidence angle


⸻

Stage 7: Inversion

Solve for reflectivity profile:

h = A† Y

Where:

A† = pseudoinverse(A)

The magnitude of h(z) produces the tomogram.

⸻

5. Visualization

For each pixel:

h(z)

represents reflectivity versus depth.

The tomographic image can be displayed as:

|h(z)|


⸻

6. Required Data

To run the algorithm we need:

Single SAR SLC scene

The scene must contain:
	•	complex phase
	•	full azimuth aperture

⸻

7. Public SAR Data Sources

The following datasets can be used.

⸻

Sentinel-1

Satellite operated by the European Space Agency.

Satellite:
	•	Sentinel‑1

Properties:

parameter	value
frequency	C-band
resolution	~5 m
orbit repeat	6–12 days

Product required:

Level-1 SLC
Mode: IW
Polarization: VV

Download sources:
	•	https://search.asf.alaska.edu
	•	https://dataspace.copernicus.eu

⸻

TerraSAR-X

Operated by the German Aerospace Center.

Satellite:
	•	TerraSAR‑X

Properties:

parameter	value
frequency	X-band
resolution	~1 m

Access requires research request.

⸻

UAVSAR

Operated by the NASA Jet Propulsion Laboratory.

Platform:
	•	UAVSAR

Advantages:
	•	high phase stability
	•	interferometric stacks

Download:

https://uavsar.jpl.nasa.gov


⸻

8. Target Scene

For reproducing the pyramid experiment use:

29.9792° N
31.1342° E

This corresponds to the Giza pyramid complex.

⸻

9. Synthetic Data Test

Before using real SAR data it is recommended to generate synthetic scenes.

Example:

scatterers at depths:
z = [0, 10, 25]

Generate simulated measurement vector:

Y = A h

Add noise:

Y_noise = Y + ε

Reconstruct:

h_est = pinv(A) Y_noise

This verifies algorithm correctness.

⸻

10. Engineering Architecture

Recommended system architecture:

SAR data loader
↓
SLC preprocessing
↓
Doppler decomposition
↓
sub-aperture reconstruction
↓
motion estimation
↓
tomography inversion
↓
visualization


⸻

11. Optimization Strategy

Key bottlenecks:
	•	Doppler decomposition
	•	per-pixel inversion
	•	pixel tracking

Recommended improvements:

Precompute inversion operator

G = (AᴴA + αI)^-1 Aᴴ

Then

h = G Y


⸻

GPU acceleration

Use:
	•	cuFFT
	•	PyTorch / CUDA
	•	batched matrix multiplications

⸻

Batched tomography

Instead of solving per pixel:

H = G Y

Where:

Y ∈ ℂ^(K × P)


⸻

Mixed precision

Use:

complex64

for real-time preview.

⸻

12. Research Questions

This project should answer:
	1.	Can the tomography pipeline be reproduced?
	2.	Do similar structures appear in different SAR datasets?
	3.	Are the results stable across satellites?
	4.	Can the inversion be accelerated to real-time?

⸻

13. Project Structure

Suggested repo structure:

sar_tomography/

docs/
research.md

data/
raw/
processed/

src/
slc_loader.py
doppler_subaperture.py
pixel_tracking.py
steering_matrix.py
tomography_solver.py
visualization.py

experiments/
sentinel_giza.ipynb
synthetic_tests.ipynb


⸻

14. Development Roadmap

Phase 1

synthetic scene simulation
verify inversion math

Phase 2

load Sentinel-1 SLC
generate sub-apertures

Phase 3

pixel motion extraction

Phase 4

tomographic reconstruction

Phase 5

GPU optimization


⸻

15. Expected Outcome

Possible outcomes:

result	interpretation
structures appear consistently	algorithm valid
structures vary by dataset	inversion artifact
no structures appear	missing processing step


⸻

16. Conclusion

The Doppler sub-aperture tomography method can be reconstructed using publicly available SAR SLC data.

The combination of:
	•	Doppler decomposition
	•	micro-motion extraction
	•	steering matrix inversion

provides a feasible path to reproduce the algorithm.

The next step is implementing the pipeline and testing it against real SAR scenes.

⸻

If you’d like, I can also produce three additional documents that would dramatically accelerate the project:
	1.	A technical deep dive on the SAR math (with derivations)
	2.	A ready-to-run Python implementation skeleton (~1200 lines)
	3.	A Sentinel-1 dataset selection guide for the pyramids

Those would make it possible to start producing real tomograms within a day or two.
