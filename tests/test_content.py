"""Content detection + background-trim: faint ink is found, background border is
removed, and content is never clipped."""
import numpy as np
import cv2

from folio.stages import content


def _page_on_black(pad=40, page=200, draw_ink=True):
    """A light-grey page centred on a black (warp-padding) border, optional ink."""
    img = np.zeros((page + 2 * pad, page + 2 * pad, 3), np.uint8)
    img[pad:pad + page, pad:pad + page] = (235, 235, 235)  # paper
    if draw_ink:
        cv2.putText(img, "abc", (pad + 20, pad + page // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (40, 40, 40), 4)
    return img, pad, page


def test_trim_removes_black_border():
    img, pad, page = _page_on_black()
    x0, y0, x1, y1 = content.trim_background_border(img)
    # the black padding should be mostly gone: kept box ~ the page, not the frame
    assert x0 >= pad - 10 and y0 >= pad - 10
    assert x1 <= pad + page + 10 and y1 <= pad + page + 10


def test_trim_never_clips_ink():
    img, pad, page = _page_on_black(draw_ink=True)
    x0, y0, x1, y1 = content.trim_background_border(img)
    ink, _ = content.content_mask(img)
    border = ink.copy()
    border[y0:y1, x0:x1] = 0
    assert not border.any(), "trim clipped detected content pixels"


def test_trim_noop_on_tight_crop():
    # a full-frame page with no background border: trim should keep ~everything
    img = np.full((220, 200, 3), 235, np.uint8)
    cv2.putText(img, "hello", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = content.trim_background_border(img)
    assert (x1 - x0) >= 0.9 * w and (y1 - y0) >= 0.9 * h


def test_faint_ink_detected():
    # a very low-contrast stroke (paper 235, ink 215) on a clean page
    img = np.full((300, 300, 3), 235, np.uint8)
    cv2.line(img, (40, 150), (260, 150), (215, 215, 215), 3)
    _, cov = content.content_mask(img)
    assert cov > 0.0, "faint stroke not detected at all"
    # and a truly blank clean page should read lower than the faint-ink one
    blank = np.full((300, 300, 3), 235, np.uint8)
    _, cov_blank = content.content_mask(blank)
    assert cov > cov_blank


def test_paper_box_excludes_dark_border_keeps_ink():
    # bright page with text, sitting in a dark "binding/scanner" frame
    full = np.full((400, 400, 3), 20, np.uint8)          # dark surround
    full[60:340, 90:330] = (230, 230, 230)               # bright paper
    cv2.putText(full, "No 338", (100, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (30, 30, 30), 3)                     # margin annotation
    box = content.paper_box(full)
    assert box is not None
    x0, y0, x1, y1 = box
    # box should sit inside the dark frame (dropped background) ...
    assert x0 >= 70 and y0 >= 40 and x1 <= 350 and y1 <= 360
    # ... but still contain all the detected ink (no marginalia clipped)
    ink, _ = content.content_mask(full)
    b = ink.copy(); b[y0:y1, x0:x1] = 0
    assert (b > 0).sum() <= 0.01 * (ink > 0).sum()


def test_paper_box_noop_when_no_background():
    img = np.full((300, 280, 3), 232, np.uint8)
    cv2.putText(img, "text", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
    assert content.paper_box(img) is None  # already full-frame paper -> keep


def test_enhance_faint_increases_contrast():
    img = np.full((200, 200, 3), 200, np.uint8)
    cv2.putText(img, "x", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (170, 170, 170), 5)
    out = content.enhance_faint(img)
    assert out.shape == img.shape
    assert out.std() >= img.std()  # contrast not reduced
