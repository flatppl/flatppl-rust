//! Emit the shared structural plan at an observed IID Normal factor.

use super::{Emitter, Ex, block, functions::FnCtx};
use flatppl_core::{NodeId, Symbol};
use flatppl_model_analysis::plan_iid_normal;
use std::collections::HashMap;

/// The bindings in scope when the ordinary observation was rendered.
pub(super) struct NormalSite {
    lhs: String,
    observed: NodeId,
    law: NodeId,
    parameters: [NodeId; 2],
    bindings: HashMap<Symbol, String>,
    conditioned: usize,
}

impl Emitter<'_> {
    pub(super) fn normal_site(
        &self,
        lhs: &str,
        observed: NodeId,
        law: NodeId,
        conditioned: usize,
    ) -> Option<NormalSite> {
        if !self.opts.sufficient_statistics {
            return None;
        }
        let iid = self.resolve(law);
        if self.head(iid) != Some("iid") || self.decls.get(lhs)?.base != "vector" {
            return None;
        }
        let normal = self.resolve(*self.call(iid)?.args.first()?);
        if self.head(normal) != Some("Normal") {
            return None;
        }
        Some(NormalSite {
            lhs: lhs.to_string(),
            observed,
            law,
            conditioned,
            parameters: self.law_args("Normal", normal).ok()?.try_into().ok()?,
            bindings: self.env.clone(),
        })
    }

    pub(super) fn observed_normal(
        &mut self,
        site: NormalSite,
        conditioned: &[Symbol],
    ) -> Option<String> {
        let NormalSite {
            lhs,
            observed,
            law,
            parameters: [mu, sigma],
            bindings,
            conditioned: n,
        } = site;
        let scoring: Vec<_> = bindings
            .iter()
            .filter(|(_, name)| self.fields.values().any(|param| param == *name))
            .map(|(&sym, _)| sym)
            .collect();
        let plan = plan_iid_normal(self.m, law, observed, &scoring, &conditioned[..n])?;
        let count = i32::try_from(plan.observation_count()).ok()?;
        // Each plan owns a module with its own synthetic names and node IDs.
        // Publish only the completed fragment and its reserved Stan names.
        let mut child = Emitter::new(plan.module(), self.opts);
        child.used.clone_from(&self.used);
        child.env = bindings;
        child.decls.clone_from(&self.decls);
        child.fctx = Some(FnCtx::default());
        let mu_value = child.real(mu).ok()?;
        let sigma_value = child.real(sigma).ok()?;
        let loc = child.fresh("normal_mu");
        let scale = child.fresh("normal_sigma");
        let mut model = child.plan_lines();
        model.push(format!("real {loc} = {};", mu_value.s));
        model.push(format!("real {scale} = {};", sigma_value.s));
        child.fctx.as_mut()?.lifted.extend([
            (mu, Ex::atom(&loc)),
            (sigma, Ex::atom(&scale)),
            (observed, Ex::atom(&lhs)),
        ]);
        child.fctx.as_mut()?.plan_arrays = Some(Vec::new());

        let usable = child.fresh(&format!("{lhs}_summarized"));
        let mut declarations = vec![format!("int {usable} = 0;")];
        let mut setup = Vec::new();
        for (kind, summary) in plan.named_summaries() {
            let slot = child.fresh(&format!("{lhs}_{kind}"));
            let shape = child.shape(summary.rhs)?;
            if shape.rank != 0 {
                return None;
            }
            let ty = if shape.int { "int" } else { "real" };
            declarations.push(format!("{ty} {slot} = 0;"));
            let value = child.expr(summary.rhs).ok()?;
            setup.extend(child.plan_lines());
            setup.push(format!("{slot} = {};", value.s));
            child.env.insert(summary.name, slot);
        }
        let finite = child.expr(plan.observations_are_finite()).ok()?;
        // For int-sized plates these bounds keep squared standardized residuals
        // and their scale derivatives below f64::MAX. Outside them Stan's
        // native density handles both validation and reverse-mode arithmetic.
        setup.push(format!("{usable} = {};", finite.s));
        let mut guarded = String::new();
        block(
            &mut guarded,
            &format!("if (num_elements({lhs}) == {count} && max(abs({lhs})) < 1e38)"),
            &setup,
        );
        declarations.push(guarded.trim_end().to_string());

        child.fctx.as_mut()?.plan_arrays = None;
        let score = child.expr(plan.score()).ok()?;
        let mut reduced = child.plan_lines();
        reduced.push(format!("target += {};", score.s));
        let mut branch = String::new();
        block(
            &mut branch,
            &format!("if ({usable} && abs({loc}) < 1e38 && {scale} > 1e-38 && {scale} < 1e38)"),
            &reduced,
        );
        // Both branches retain the same normalization when Stan uses propto.
        block(
            &mut branch,
            "else",
            &[format!("target += normal_lpdf({lhs} | {loc}, {scale});")],
        );
        model.push(branch.trim_end().to_string());
        if !child.data.is_empty()
            || !child.tdata.is_empty()
            || !child.locals.is_empty()
            || !child.tparams.is_empty()
            || !child.functions.is_empty()
        {
            return None;
        }
        self.used = child.used;
        self.tdata.extend(declarations);
        let mut out = String::new();
        block(&mut out, "", &model);
        Some(out.trim().to_string())
    }

    fn plan_lines(&mut self) -> Vec<String> {
        std::mem::take(&mut self.fctx.as_mut().expect("plan expression scope").lines)
    }
}
