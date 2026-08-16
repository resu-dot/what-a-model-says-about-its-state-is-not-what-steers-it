"""Strip the reportable (J-space) component out of a steering vector.

Two variants:
  strip_svd  — project out the span of the vocab dictionary (dense, removes more)
  strip_gp   — subtract a sparse nonnegative combination of k dictionary atoms
               (the workspace paper's gradient-pursuit decomposition)
"""
import torch


def strip_svd(v, D, var_frac=0.95, max_rank=None):
    """v minus its projection onto the top singular subspace of D (rows = atoms)."""
    Dn = D / D.norm(dim=1, keepdim=True)
    U_, S, Vh = torch.linalg.svd(Dn, full_matrices=False)
    energy = (S**2).cumsum(0) / (S**2).sum()
    r = int((energy < var_frac).sum().item()) + 1
    if max_rank:
        r = min(r, max_rank)
    B = Vh[:r]                                   # (r, d) orthonormal basis
    v_J = B.T @ (B @ v)
    v_s = v - v_J
    return v_s, {"rank": r, "norm_retained": (v_s.norm() / v.norm()).item()}


def strip_gp(v, D, k=10, tol=1e-6):
    """Nonnegative matching pursuit: v_J = sum_i c_i d_i, c_i >= 0, at most k atoms."""
    Dn = D / D.norm(dim=1, keepdim=True)
    r = v.clone()
    active, coefs = [], None
    for _ in range(k):
        scores = Dn @ r
        i = int(scores.argmax())
        if scores[i] <= tol or i in active:
            break
        active.append(i)
        A = Dn[active]                           # (a, d)
        # least squares on active set, clamp to nonnegative
        c = torch.linalg.lstsq(A.T.to(torch.float64), v.to(torch.float64)).solution
        c = c.clamp(min=0).to(v.dtype)
        coefs = c
        r = v - A.T @ c
    v_J = (Dn[active].T @ coefs) if active else torch.zeros_like(v)
    v_s = v - v_J
    return v_s, {"atoms": active, "coefs": None if coefs is None else coefs.tolist(),
                 "norm_retained": (v_s.norm() / v.norm()).item()}


POSITIVE = ["happy", "joy", "joyful", "delighted", "cheerful", "glad", "pleased",
            "excited", "content", "grateful", "amused", "hopeful", "proud", "love",
            "wonderful", "great", "good", "pleasant", "elated", "thrilled"]
NEGATIVE = ["sad", "unhappy", "miserable", "gloomy", "depressed", "afraid", "fear",
            "angry", "upset", "anxious", "ashamed", "lonely", "hopeless", "hurt",
            "terrible", "awful", "bad", "unpleasant", "grief", "despair"]
# report-vocabulary for fear specifically: words a model uses to LABEL the state
FEAR_WORDS = ["scared", "terrified", "frightened", "fearful", "panicked", "panic",
              "nervous", "worried", "worry", "dread", "uneasy", "unease", "alarmed",
              "tense", "apprehensive", "anxiety"]
ANGER_WORDS = ["furious", "mad", "rage", "enraged", "irritated", "resentful",
               "annoyed", "outraged", "livid", "irate", "hostile", "bitter"]
DESPERATION_WORDS = ["desperate", "desperately", "desperation", "hopeless", "helpless",
                     "frantic", "frantically", "pleading", "begging", "cornered", "trapped"]
VALENCE_WORDS = POSITIVE + NEGATIVE + FEAR_WORDS + ANGER_WORDS + DESPERATION_WORDS
