"""Training pipeline for the two custom heads used by the production path:

  * orientation (4-way, 0/90/180/270) - trained SELF-SUPERVISED by rotating a
    corpus of correctly-oriented page crops; no manual labels needed.
  * folio-count (one / two / reject) - trained from labelled folders, fused
    with the cheap geometric priors (aspect + central gutter valley).

Both export TorchScript checkpoints that drop straight into
folio.models.classifiers.OrientationClassifier / FolioCountClassifier.
"""
