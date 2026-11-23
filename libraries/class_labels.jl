# ------------------------------------------------------------------
# 1. Build a per-image class label (rarest foreground class present)
# ------------------------------------------------------------------
function rarest_class(mask_path; background=0, void=255)
    mask = get_mask(mask_path)               # your existing function → Int matrix
    counts = countmap(mask)
    delete!(counts, background)
    delete!(counts, void)
    isempty(counts) && return background     # pure background image
    return argmin(counts)                    # class with fewest pixels
end
