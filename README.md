# AnalogSAGE Framework

AnalogSAGE is organized so that each module and stage is separated into an individual file. These files can be called through functions and flexibly composed to test different modules individually.

## Environment Setup

You need to configure the following according to your local device environment:

* `PySpice`
* `npspice`
* The corresponding `PDK`

## Data Preparation

Before running the full flow, pre-generate the following:

* **Design knowledge RAG**
* **Topology vector database**
