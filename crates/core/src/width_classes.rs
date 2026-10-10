//! Width classes for packing rows of unequal length into padded blocks. The
//! pyhf converter's normsys lane tensor and sample yields and the emitter's
//! segment reductions share one partition so they cannot drift.

/// Padding share above which rows split into another width class.
/// Padding costs work on CPU and each class costs kernel launches on GPU.
const MAX_PADDING: f64 = 0.15;

/// Smallest width of the form 2^k or 3 * 2^k that holds `length` lanes.
/// The pairwise product tree then halves without an odd tail, except once at
/// width 3, and each odd tail costs a slice and a concatenate that split GPU
/// fusions.
fn tree_width(length: usize) -> usize {
    let power = length.next_power_of_two();
    if power / 4 * 3 >= length {
        power / 4 * 3
    } else {
        power
    }
}

/// Width of a class whose widest row has `length` lanes. A lone row needs no
/// common width: padding it would repeat its parameter in alpha, and repeated
/// indices turn the alpha gradient from a concatenate into a scatter.
pub fn class_width(length: usize, rows: usize) -> usize {
    if rows == 1 {
        length
    } else {
        tree_width(length)
    }
}

/// Fewest contiguous classes over `lengths`, sorted in decreasing order, whose
/// rows padded to each class's width keep the lanes beyond each row's own
/// tree width under [`MAX_PADDING`]. One row per class meets the bound, so
/// the search ends. Returns the end of each class.
pub fn width_classes(lengths: &[usize]) -> Vec<usize> {
    // Tree rounding alone can exceed the bound, as one row per class shows.
    let floor: usize = lengths.iter().map(|&length| tree_width(length)).sum();
    padded_classes(lengths.len(), floor, |rows| {
        class_width(lengths[rows.start], rows.len()) * rows.len()
    })
}

/// Fewest contiguous classes over `count` items whose total `lanes` exceed
/// the unpadded `floor` by at most [`MAX_PADDING`] of the lanes. `lanes` gives
/// one class's padded lanes and must equal the floor's share for one item.
/// Returns the end of each class.
pub fn padded_classes(
    count: usize,
    floor: usize,
    lanes: impl Fn(std::ops::Range<usize>) -> usize,
) -> Vec<usize> {
    // Fewest lanes covering the first i items with the current class count.
    let mut best: Vec<Option<(usize, Vec<usize>)>> = vec![None; count + 1];
    best[0] = Some((0, Vec::new()));
    loop {
        if let Some((total, ends)) = &best[count]
            && total.saturating_sub(floor) as f64 <= MAX_PADDING * *total as f64
        {
            return ends.clone();
        }
        best = (0..=count)
            .map(|i| {
                (0..i)
                    .filter_map(|j| {
                        let (total, ends) = best[j].as_ref()?;
                        Some((total + lanes(j..i), ends))
                    })
                    .min_by_key(|(total, _)| *total)
                    .map(|(total, ends)| (total, [ends.as_slice(), &[i]].concat()))
            })
            .collect();
    }
}
