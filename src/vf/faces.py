"""Face recognisers (insightface). A = Arc2Face's ArcFace (defines the identity space);
B = an independent recogniser used only for evaluation (paper Sec. IV-L)."""
import os
import numpy as np

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def build_apps(arc_root, indep_root="models_indep"):
    from insightface.app import FaceAnalysis
    a = FaceAnalysis(name="antelopev2", root=arc_root, providers=PROVIDERS)
    a.prepare(ctx_id=0, det_size=(640, 640))
    b = FaceAnalysis(name="buffalo_l", root=os.path.abspath(indep_root), providers=PROVIDERS,
                     allowed_modules=["detection", "recognition"])
    b.prepare(ctx_id=0, det_size=(640, 640))
    return a, b


def embed_largest(app, bgr):
    """Unit-norm embedding of the largest detected face, or (None, 0.0) if none found."""
    faces = app.get(bgr)
    if not faces:
        return None, 0.0
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    e = f.embedding.astype(np.float32)
    return e / np.linalg.norm(e), float(f.det_score)
