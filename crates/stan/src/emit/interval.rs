//! `T[lo, hi]` with both ends finite, as the law's statement minus its log
//! interval mass from a helper (ruling of 2026-11-18). Stan's `T[lo, hi]`
//! takes log(F(hi) − F(lo)) from the `lcdf` pair, which rounds to 0 − 0 far
//! in the upper tail. The helper switches to the survival pair there
//! (`tail: lccdf` rows) or, for a law whose `lccdf` is not exact far out, to
//! the `lcdf` pair of the mirror image (`tail: mirror_lcdf`, symmetric laws).
//! The flag is set only for laws whose far-tail functions were checked in
//! BridgeStan against scipy; every other law keeps `T[lo, hi]`.

use super::Emitter;
use super::mixture::split_args;
use super::structure::{LOOP, Law};
use crate::refuse::Result;

impl<'m> Emitter<'m> {
    /// The statements for `lhs ~ law` when `law` is truncated on both sides
    /// and its row has a `tail` flag; `None` keeps `T[lo, hi]`.
    pub(super) fn interval_mass(
        &mut self,
        lhs: &str,
        law: &Law,
        lhs_rank: usize,
        per_element: bool,
    ) -> Result<Option<Vec<String>>> {
        let Some((Some(lo), Some(hi))) = &law.trunc else {
            return Ok(None);
        };
        // The mass is shared by every element only with shared arguments.
        // A discrete law's ends are integer-typed here (`normalize` refuses others).
        if law.tail.is_empty()
            || law.multivariate
            || per_element
            || lhs_rank > 1
            || law.lhs("{y}") != "{y}"
            || law.call.contains(LOOP)
        {
            return Ok(None);
        }
        let open = law.call.find('(').unwrap_or(law.call.len());
        let stan = law.call[..open].to_string();
        let args = law.call[open..]
            .strip_prefix('(')
            .and_then(|x| x.strip_suffix(')'))
            .map(split_args)
            .unwrap_or_default();
        let helper = self.interval_helper(&stan, law, args.len());
        let mut call_args = vec![lo.clone(), hi.clone()];
        call_args.extend(args);
        let mass = format!("{helper}({})", call_args.join(", "));
        let count = if lhs_rank == 0 {
            String::new()
        } else {
            format!("{} * ", self.extent_text_of(lhs))
        };
        let statement = format!("{lhs} ~ {};", law.call);
        let minus = format!("target += -{count}{mass};");
        if !self.is_data(lhs) {
            // A parameter's declared bounds keep it in [lo, hi].
            return Ok(Some(vec![statement, minus]));
        }
        // Data outside [lo, hi] has zero density; inside, the mass is finite.
        Ok(self
            .guarded(lhs, Some(lo), Some(hi), vec![statement, minus])
            .map(|g| vec![g]))
    }

    /// The name of the helper for the Stan law `stan`, emitted once.
    fn interval_helper(&mut self, stan: &str, law: &Law, nargs: usize) -> String {
        let name = format!("{stan}_interval_lmass");
        if self
            .functions
            .iter()
            .any(|f| f.contains(&format!(" {name}(")))
        {
            return name;
        }
        let ty = if law.discrete { "int" } else { "real" };
        let params: Vec<String> = (1..=nargs).map(|k| format!("p{k}")).collect();
        let decl: String = params.iter().map(|p| format!(", real {p}")).collect();
        let p = params.join(", ");
        let at = |x: &str| {
            if p.is_empty() {
                x.to_string()
            } else {
                format!("{x} | {p}")
            }
        };
        let text = if law.tail == "mirror_lcdf" {
            // Symmetric about the first parameter (the location).
            let mu = params.first().cloned().unwrap_or_else(|| "0".into());
            format!(
                "real {name}({ty} lo, {ty} hi{decl}) {{\n  \
                 // Above the median the upper tail comes from its mirror image: {stan}_lccdf is -inf far out.\n  \
                 real la = {stan}_lcdf({});\n  \
                 if (la > log(0.5)) {{\n    \
                 return log_diff_exp({stan}_lcdf({}), {stan}_lcdf({}));\n  \
                 }}\n  \
                 return log_diff_exp({stan}_lcdf({}), la);\n}}",
                at("lo"),
                at(&format!("2 * {mu} - lo")),
                at(&format!("2 * {mu} - hi")),
                at("hi"),
            )
        } else {
            // A discrete law's mass counts `lo`; a continuous one clamps both
            // ends to its support.
            let floor = match law.floor_arg {
                Some(k) => Some(format!("p{}", k + 1)),
                None => law.bounds.0.clone().filter(|b| b.parse::<f64>().is_ok()),
            };
            let (a, b, ends) = match (law.discrete, &floor) {
                (true, _) => ("a", "hi", "  int a = lo - 1;\n".to_string()),
                (false, Some(f)) => (
                    "a",
                    "b",
                    format!("  real a = fmax(lo, {f});\n  real b = fmax(hi, {f});\n"),
                ),
                (false, None) => ("lo", "hi", String::new()),
            };
            format!(
                "real {name}({ty} lo, {ty} hi{decl}) {{\n{ends}  \
                 // Above the median the survival pair keeps the precision that the lcdf pair loses.\n  \
                 real la = {stan}_lccdf({});\n  \
                 if (la < log(0.5)) {{\n    \
                 return log_diff_exp(la, {stan}_lccdf({}));\n  \
                 }}\n  \
                 return log_diff_exp({stan}_lcdf({}), {stan}_lcdf({}));\n}}",
                at(a),
                at(b),
                at(b),
                at(a),
            )
        };
        self.functions.push(text);
        name
    }

    /// The declared outer extent of a Stan variable, as `extent_of` prints it.
    fn extent_text_of(&self, name: &str) -> String {
        self.decls
            .get(name)
            .and_then(|t| t.array.first().or(t.dims.first()).cloned())
            .unwrap_or_else(|| format!("num_elements({name})"))
    }
}
