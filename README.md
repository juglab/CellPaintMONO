# CellPaintMONO
Tools for analysis and comparison of cell painting data.

To install:
```
git clone https://github.com/juglab/CellPaintMONO.git
cd CellPaintMONO
uv sync
```

Welcome to Cell Paint Mono! This library provides the workflow and tools for cell painting data analysis and comparison between original data and MicroSplit-predicted data.

### What is Cell Painting?

Cell Painting is a morphological profiling assay that multiplexes fluorescent dyes, imaged in multiple channels, to reveal a specified number of broadly relevant cellular components or organelles. Cells are plated in multiwell plates, perturbed with the treatments to be tested, stained, fixed, and imaged on a high-throughput microscope.
There are a growing number of options for the next step of image analysis (discussed below) where the goal is to identify cells and extract their morphological features in order to infer more information about the samples and finally produce a complex phenotypic profile. You can read more about cell painting [here](https://www.nature.com/articles/nprot.2016.105)

There are multiple groups and consortia involved in producing open [FAIR](https://www.nature.com/articles/s41592-023-01987-9) (findable, accessible, interoperable and reusable) cell painting datasets, including the [JUMP  consortium](https://europepmc.org/article/ppr/ppr635627) (Joint Undertaking for Morphological Profiling) and [EU-openscreen](https://www.sciencedirect.com/science/article/pii/S2589004225007060). All of the cell painting data published is available via Cell Painting Gallery hosted on [AWS](https://cellpainting-gallery.s3.amazonaws.com/index.html). You can read more about the gallery and how to navigate it [here](https://www.nature.com/articles/s41592-024-02399-z)

### What is MicroSplit?

Fluorescence microscopy is a key step in the cell painting assay but it faces technical limitations which in turn require trade-offs in imaging speed, resolution, and depth. Furthermore, in the context of cell painting, the image acquisition step requires a lot of resources (equipment, reagents, computational storage) and imaging time... that's where we come in.

MicroSplit is a computational multiplexing technique based on deep learning that allows multiple cellular structures to be imaged in a single fluorescent channel and then unmix them by computational means, allowing faster imaging and reduced photon exposure. The [MicroSplit paper](https://www.biorxiv.org/content/10.1101/2025.02.10.637323v3.abstract) shows how it efficiently separates superimposed noisy structures into distinct denoised fluorescent image channels. In our example notebooks, we demonstrate how MicroSplit effectively unmixes up to 5 channels from a single superimposed image and subsequently performs for image analysis in comparison to the original cell painting data.

MicroSplit is implemented in the [CAREamics library](https://careamics.github.io/0.1/) with installation instructions available [here](https://careamics.github.io/0.1/installation/)

### What does this mean?

Using MicroSplit, cell painting assays can be revised to enable multiple cellular structures to be imaged in a single fluorescent channel and then computationally unmixed before image analysis steps.

### How do we analyse the data?

After image acquisition, cell painting images are typically pre-processed with software such as [CellProfiler](https://cellprofiler.org/) or [DeepProfiler](https://github.com/cytomining/DeepProfiler). The data must be collated for profile processing, which may be done using [Pycytominer](https://pycytominer.readthedocs.io/en/v1.0.0/), a growing number of alternative libraries or custom functions. This enables the creation of the final profiles which can be used for a variety of downstream applications, drug discovery & compound screening, multimodal data integration, disease research, etc.

For the purposes of demonstrating the efficacy of Microsplit, we compare how the Microsplit-predicted cell painting data performs in comparison to the original data from pre-processing to profile creation. We use CellProfiler v.4.2.8 to run two analysis pipelines, adapted from the [cpg0000-jump-pilot experiment pipelines](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1/tree/56845c7d4dc322652952783d91dae0ffef47829f/pipelines/2020_11_04_CPJUMP1):

We compare the original vs. MicroSplit-predicted CellProfiler outputs before profile processing. We also compare the original vs. MicroSplit-predicted consensus and aggregate profiles.

The example notebooks contain the workflow and comparison analysis for the following datasets:
1. cpg0000-jump-pilot (JUMP Pilot experiment testing different perturbation conditions, different cell types, 5 channels)
2. cpg0016 (JUMP principal dataset containing 116k chemical and ~22k gene perturbations, U2OS cell line, 5 channels)
    - ORF subset
    - CRISPR subset
3. cpg0036-EU-OS-bioactives (EU-OPENSCREEN experiment with 2464 annotated bioactive compounds, HepG2 and U2OS cell line, 4 channels)

### Additional resources:
1. For JUMP documentation and examples, check out [JUMP Hub](https://broadinstitute.github.io/jump_hub/)
2. For help with CellProfiler, check out the [software documentation](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.0.4/index.html)
3. Learn more about [CellProfiler 4](https://link.springer.com/article/10.1186/s12859-021-04344-9)
4. Learn more about the [JUMP cell painting assay](https://www.nature.com/articles/s41596-023-00840-9)
