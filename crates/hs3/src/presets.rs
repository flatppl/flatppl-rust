//! parameter_points / domains -> FlatPPL preset bindings (03-value-types.md).
use crate::builder::Builder;
use crate::model::{Domain, ParameterPoint};
use flatppl_core::NodeId;
use flatppl_core::node::{Call, CallHead, NamedArg, NamedKind, Node};

/// `parameter_points` entry -> `name = record(p = v, ..., q = fixed(w))`.
pub fn emit_parameter_point(b: &mut Builder, pp: &ParameterPoint) {
    let mut fields = Vec::new();
    for e in &pp.entries {
        let mut val = b.lit_real(e.value);
        if e.r#const {
            val = b.call("fixed", &[val]);
        }
        let name = b.m.intern(&e.name);
        fields.push(NamedArg {
            kind: NamedKind::Field,
            name,
            value: val,
        });
    }
    let head = b.m.intern("record");
    let rec = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Vec::new().into(),
        named: fields.into(),
        inputs: None,
    }));
    b.bind(&pp.name, rec);
}

/// `domains` (product_domain) -> `name = cartprod(p = interval(min, max), ...)`.
/// An omitted bound (RooFit emits one-sided ranges for unbounded parameters)
/// becomes `±inf`, so `[0, ∞)` lowers to `interval(0.0, inf)`.
pub fn emit_domain(b: &mut Builder, d: &Domain) {
    let mut fields = Vec::new();
    for ax in &d.axes {
        let lo = bound_node(b, ax.min, false);
        let hi = bound_node(b, ax.max, true);
        let interval = b.call("interval", &[lo, hi]);
        let name = b.m.intern(&ax.name);
        fields.push(NamedArg {
            kind: NamedKind::Field,
            name,
            value: interval,
        });
    }
    let head = b.m.intern("cartprod");
    let cp = b.m.alloc(Node::Call(Call {
        head: CallHead::Builtin(head),
        args: Vec::new().into(),
        named: fields.into(),
        inputs: None,
    }));
    b.bind(&d.name, cp);
}

/// An interval bound node: the literal value, or `±inf` when RooFit omitted the
/// bound (an unbounded parameter range; see [`crate::model::DomainAxis`]).
/// `positive` selects `+inf` (upper) vs `-inf` (lower).
fn bound_node(b: &mut Builder, v: Option<f64>, positive: bool) -> NodeId {
    match v {
        Some(x) => b.lit_real(x),
        None => {
            let inf_sym = b.m.intern("inf");
            let inf = b.m.alloc(Node::Const(inf_sym));
            if positive {
                inf
            } else {
                b.call("neg", &[inf])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Builder;
    use crate::model::{DomainAxis, ParamValue, ParameterPoint};
    use flatppl_syntax::{Syntax, print_with};

    #[test]
    fn parameter_point_to_record() {
        let mut m = flatppl_core::Module::new();
        let pp = ParameterPoint {
            name: "default".into(),
            entries: vec![
                ParamValue {
                    name: "mu_param".into(),
                    value: 5.28,
                    r#const: false,
                },
                ParamValue {
                    name: "sigma_param".into(),
                    value: 0.003,
                    r#const: false,
                },
            ],
        };
        {
            let mut b = Builder::new(&mut m);
            emit_parameter_point(&mut b, &pp);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("record"), "got: {text}");
        assert!(text.contains("mu_param"), "got: {text}");
        assert!(text.contains("5.28"), "got: {text}");
    }

    #[test]
    fn const_entry_wraps_fixed() {
        let mut m = flatppl_core::Module::new();
        let pp = ParameterPoint {
            name: "nominal".into(),
            entries: vec![
                ParamValue {
                    name: "alpha".into(),
                    value: 1.0,
                    r#const: true,
                },
                ParamValue {
                    name: "beta".into(),
                    value: 2.0,
                    r#const: false,
                },
            ],
        };
        {
            let mut b = Builder::new(&mut m);
            emit_parameter_point(&mut b, &pp);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("record"), "got: {text}");
        assert!(text.contains("fixed"), "got: {text}");
        assert!(text.contains("alpha"), "got: {text}");
        assert!(text.contains("1"), "got: {text}");
        // beta should NOT be wrapped in fixed
        assert!(text.contains("beta"), "got: {text}");
    }

    #[test]
    fn domain_to_cartprod() {
        let mut m = flatppl_core::Module::new();
        let d = Domain {
            name: "phase_space".into(),
            axes: vec![
                DomainAxis {
                    name: "mass".into(),
                    min: Some(5.0),
                    max: Some(6.0),
                },
                DomainAxis {
                    name: "pt".into(),
                    min: Some(0.0),
                    max: Some(100.0),
                },
                // RooFit-style one-sided range (no upper bound) → interval(0.0, inf).
                DomainAxis {
                    name: "eta".into(),
                    min: Some(0.0),
                    max: None,
                },
            ],
        };
        {
            let mut b = Builder::new(&mut m);
            emit_domain(&mut b, &d);
        }
        let text = print_with(&m, Syntax::Minimal);
        assert!(text.contains("cartprod"), "got: {text}");
        assert!(text.contains("interval"), "got: {text}");
        assert!(text.contains("mass"), "got: {text}");
        assert!(text.contains("pt"), "got: {text}");
        assert!(text.contains("5"), "got: {text}");
        assert!(text.contains("100"), "got: {text}");
        // The unbounded axis lowers to an `inf` upper bound.
        assert!(text.contains("eta"), "got: {text}");
        assert!(text.contains("inf"), "unbounded axis must use inf, got: {text}");
    }
}
