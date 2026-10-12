use crate::{Error, GroupLevel, Result};
use std::collections::HashSet;

/// brms uses make.names(unique=TRUE), then removes dots and underscores.
pub(crate) fn predictor_names(levels: &[GroupLevel]) -> Result<Vec<String>> {
    let raw: Vec<_> = levels
        .iter()
        .skip(1)
        .map(|level| {
            let label = match level {
                GroupLevel::Real(value) => real_label(*value),
                _ => level.to_string(),
            };
            if !label.is_ascii() {
                return Err(Error("categorical response labels must be ASCII".into()));
            }
            let name = format!(
                "mu{}",
                label
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() || c == '_' {
                        c
                    } else {
                        '.'
                    })
                    .collect::<String>()
            );
            let changed = name != format!("mu{label}");
            Ok((name, changed))
        })
        .collect::<Result<_>>()?;
    let reserved: HashSet<_> = raw.iter().map(|(name, _)| name.clone()).collect();
    // make.names preserves already-valid names before resolving converted ones.
    let mut order: Vec<_> = (0..raw.len()).collect();
    order.sort_by_key(|&i| raw[i].1);
    let mut used = HashSet::new();
    let mut names = HashSet::new();
    let mut result = vec![String::new(); raw.len()];
    for i in order {
        let name = &raw[i].0;
        let mut unique = name.clone();
        let mut suffix = 1;
        if used.contains(&unique) {
            loop {
                unique = format!("{name}.{suffix}");
                suffix += 1;
                if !reserved.contains(&unique) && !used.contains(&unique) {
                    break;
                }
            }
        }
        used.insert(unique.clone());
        let name = unique.replace(['.', '_'], "");
        if name == "mu" {
            return Err(Error(
                "nonreference categorical labels must contain an ASCII letter or digit".into(),
            ));
        }
        if !names.insert(name.clone()) {
            return Err(Error("categorical response labels have colliding brms parameter names; rename the categories".into()));
        }
        result[i] = name;
    }
    Ok(result)
}

// R's as.character uses 15 significant digits and prefers fixed notation
// when it is no wider than scientific notation (the default scipen = 0).
fn real_label(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let magnitude = value.abs();
    let mut shift = magnitude.log10().floor() as i32 - 14;
    let mut scaled = if shift.abs() > 22 {
        if shift <= -308 {
            (magnitude * 1e303) / 10_f64.powf(f64::from(shift + 303))
        } else {
            magnitude / 10_f64.powf(f64::from(shift))
        }
    } else if shift >= 0 {
        magnitude / 10_f64.powi(shift)
    } else {
        magnitude * 10_f64.powi(-shift)
    };
    if scaled < 1e14 {
        scaled *= 10.0;
        shift -= 1;
    }
    let mut rounded = scaled.round_ties_even();
    let mut digits = 15;
    while digits > 0 {
        rounded /= 10.0;
        if rounded.fract() != 0.0 {
            break;
        }
        digits -= 1;
    }
    if digits == 0 {
        digits = 1;
        shift += 1;
    }
    let exponent = shift + 14;
    let decimals = (digits - exponent - 1).max(0) as usize;
    let fixed = format!("{value:.decimals$}");
    let scientific = format!("{value:.precision$e}", precision = digits as usize - 1);
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let scientific = format!("{mantissa}e{exponent:+03}");
    if fixed.len() <= scientific.len() {
        fixed
    } else {
        scientific
    }
}
